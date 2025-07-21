# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import json
import os
import shutil
import struct
import tempfile

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def code(dtype):
    for k in dtypes:
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITFADS"
HDR_SIZE = 24  # bytes


class FA3Dataset(IterableDataset):
    def __init__(
        self, filenames, n_chunks, block_size, seed=12345, shuffle=True, wrap=False, num_processes=1, process_rank=0
    ):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        return FA3DatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
        )


class FA3DatasetBuilder(object):
    def __init__(self, outdir, prefix, chunk_size, max_seq_len, dtype="auto", vocab_size=None):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        self._max_seq_len = max_seq_len
        self._outdir = outdir
        self._prefix = prefix
        self.reset_arrays()
        self._version = 1
        self._total_tokens = 0
        self._filenames = []

    def reset_arrays(self):
        self._arr = []
        self._arr_lens = []
        self._arr_ids = []
        self._arr_pos_start = []

    def _write_chunk(self):
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)

        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(np.concatenate(self._arr, dtype=self.dtype).tobytes(order="C"))

        with open(filename.replace(".bin", ".meta"), "w") as f:
            json.dump(
                {
                    "lens": self._arr_lens,
                    "ids": self._arr_ids,
                    "pos_start": self._arr_pos_start,
                    "samples": [sample.tolist() for sample in self._arr],
                },
                f,
                indent=2,
            )  # FIXME remove tokens, they are just debug

        self._filenames.append(filename)
        self._counter += 1
        self.reset_arrays()

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr, id, pos_start=0):
        # split the array into smaller chunks if it's too long
        if arr.shape[0] > self._max_seq_len:
            for i in range(0, arr.shape[0], self._max_seq_len):
                self.add_array(arr[i : i + self._max_seq_len], f"{id}_{i}", pos_start=i)
            return
        if sum(self._arr_lens) + arr.shape[0] > self._chunk_size:
            self._write_chunk()
        self._arr.append(arr)
        self._arr_lens.append(arr.shape[0])
        self._arr_ids.append(id)
        self._arr_pos_start.append(pos_start)
        self._total_tokens += arr.shape[0]

    def write_reminder(self):
        print(f"total tokens: {self._total_tokens}")
        if len(self._arr) > 0:
            self._write_chunk()


class FA3DatasetIterator:
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._sample_order = None
        self._log_file = "data/datapoints.log"  # TODO

        self._wrap = wrap

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None
        self._sizes_of_samples = []

        self._mmaps = []
        self._buffers = []
        self._meta_data = []

        self._sample_order = []
        self._curr_sample_idx_within_loaded_files = 0
        self._tmp_files = []

        self._load_n_chunks()

    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, f"File doesn't match expected format. magic: {HDR_MAGIC}, got: {magic}"
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _cleanup_tmp_files(self):
        # Close and remove all temporary files from the previous invocation
        for tmp_file in self._tmp_files:
            tmp_file.close()
            try:
                os.remove(tmp_file.name)  # Clean up the temp file from the filesystem
            except FileNotFoundError:
                pass
        self._tmp_files = []
        self._meta_data = []
        self._sample_order = []

    def _load_n_chunks(self):
        # Cleanup from the previous invocation
        self._cleanup_tmp_files()
        self._mmaps = []
        self._buffers = []

        self._sizes_of_samples = []

        if self._n_chunks > len(self._filenames[self._file_idx :]):
            if not self._wrap:  # TODO remove me
                raise StopIteration
            self._file_idx = 0

        total_blocks = 0
        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]

            # Create a NamedTemporaryFile that will persist
            temp_file = tempfile.NamedTemporaryFile(delete=False, dir="/tmp/shm/")  # We won't delete it immediately
            meta_data = json.load(open(filename.replace(".bin", ".meta"), "r"))

            # Copy the contents of the original file into the temporary file
            with open(filename, "rb") as f:
                shutil.copyfileobj(f, temp_file)

            temp_file.flush()  # Ensure all content is written
            temp_file.seek(0)  # Seek back to the start of the file

            # Persist the temp file object for later cleanup
            self._tmp_files.append(temp_file)

            # Use the temp file path with np.memmap
            temp_file_path = temp_file.name

            # Use the temp file object to read the header
            if self._dtype is None:
                self._dtype, chunk_size = self._read_header(filename)
            else:
                _, chunk_size = self._read_header(filename)

            self._sizes_of_samples.extend(meta_data["lens"])

            # Memory-map the temporary file (skipping the header)
            mmap = np.memmap(temp_file_path, mode="r", order="C", offset=HDR_SIZE, dtype=self._dtype)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))
            self._meta_data.append(meta_data)

            total_blocks += len(meta_data["ids"])

        self._file_idx += self._n_chunks
        self._sample_order = self._rng.permutation(total_blocks) if self._shuffle else range(total_blocks)

        self._curr_sample_idx_within_loaded_files = 0

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_sample_idx_within_loaded_files >= len(self._sample_order):
            self._load_n_chunks()
        # FIXME, remove all unnecessary stuff and load from jsonl directly
        cum_samples_across_files = np.cumsum([0] + [len(meta["ids"]) for meta in self._meta_data])
        file_idx_to_load_from = (
            np.searchsorted(cum_samples_across_files, self._curr_sample_idx_within_loaded_files, side="right") - 1
        )

        curr_sample_idx_within_file = (
            self._curr_sample_idx_within_loaded_files - cum_samples_across_files[file_idx_to_load_from]
        )
        buffer = self._buffers[file_idx_to_load_from]
        offset_within_buffer = np.dtype(self._dtype).itemsize * sum(
            self._sizes_of_samples[:curr_sample_idx_within_file]
        )
        # TODO log samples
        # print(
        #    f"{file_idx_to_load_from=}, {curr_sample_idx_within_file=}, {cum_samples_across_files=},{len(self._sample_order)=}, {offset_within_buffer=}, {len(buffer)=}, {self._meta_data[file_idx_to_load_from]['lens'][self._curr_sample_idx_within_loaded_files - cum_samples_across_files[file_idx_to_load_from]]=}"
        # )

        if offset_within_buffer > len(buffer):
            pass  # FIXME how is this even possible??? wrong dtype? actually can't be. lädt flüssig weiter
        # arr = np.frombuffer(
        #    buffer,
        #    dtype=self._dtype,
        #    count=self._meta_data[file_idx_to_load_from]["lens"][
        #        self._curr_sample_idx_within_loaded_files - cum_samples_across_files[file_idx_to_load_from]
        #    ],
        #    offset=offset_within_buffer,
        # )
        self._curr_sample_idx_within_loaded_files += 1

        # return torch.from_numpy(arr.astype(np.int64)), self._meta_data[file_idx_to_load_from]["samples"][
        #    curr_sample_idx_within_file
        # ]

        return {
            "tokens": torch.tensor(self._meta_data[file_idx_to_load_from]["samples"][curr_sample_idx_within_file]),
            "len": self._meta_data[file_idx_to_load_from]["lens"][curr_sample_idx_within_file],
            "pos_start": self._meta_data[file_idx_to_load_from]["pos_start"][curr_sample_idx_within_file],
        }
