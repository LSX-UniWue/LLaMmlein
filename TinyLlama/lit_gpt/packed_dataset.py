# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import os
import random
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


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
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

        return PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
        )


class PackedDatasetBuilder(object):
    def __init__(self, outdir, prefix, chunk_size, sep_token, dtype="auto", vocab_size=None):
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
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames = []

    def _write_chunk(self):
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)

        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        self._filenames.append(filename)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr):
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_reminder(self):
        if self._idx > 0:
            filename = f"{self._prefix}_{self._counter:010d}.bin"
            filename = os.path.join(self._outdir, filename)

            final_chunk = self._arr[: self._idx]

            print(f"Original length of final chunk: {len(final_chunk)}")

            chunk_size = 2048
            pad_length = chunk_size - (len(final_chunk) % chunk_size)

            if pad_length < chunk_size:
                padded_chunk = np.concatenate([final_chunk, np.full(pad_length, self._sep_token, dtype=self._dtype)])
            else:
                padded_chunk = final_chunk

            print(f"Length of final chunk after padding: {len(padded_chunk)}")

            with open(filename, "wb") as f:
                f.write(HDR_MAGIC)
                f.write(struct.pack("<Q", self._version))
                f.write(struct.pack("<B", code(self._dtype)))
                f.write(struct.pack("<Q", len(padded_chunk)))  # Write the length of the padded chunk
                f.write(padded_chunk.tobytes(order="C"))

            self._filenames.append(filename)
            self._counter += 1
            self._arr.fill(self._sep_token)
            self._idx = 0


class PackedDatasetIterator:
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._wrap = wrap

        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None
        self._n_blocks_list = []

        self._mmaps = []
        self._buffers = []

        self._block_idxs = []
        self._curr_idx = 0
        self._tmp_files = []

        self._load_n_chunks()

    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
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

    def _load_n_chunks(self):
        # Cleanup from the previous invocation
        self._cleanup_tmp_files()
        self._mmaps = []
        self._buffers = []
        self._n_blocks_list = []

        if self._n_chunks > len(self._filenames[self._file_idx :]):
            self._file_idx = 0

        total_blocks = 0
        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]

            # Create a NamedTemporaryFile that will persist
            temp_file = tempfile.NamedTemporaryFile(delete=False, dir="/dev/shm/")  # We won't delete it immediately

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

            n_blocks = chunk_size // self._block_size
            self._n_blocks_list.append(n_blocks)

            # Memory-map the temporary file (skipping the header)
            mmap = np.memmap(temp_file_path, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

            total_blocks += n_blocks

        self._file_idx += self._n_chunks
        self._block_idxs = self._rng.permutation(total_blocks) if self._shuffle else range(total_blocks)

        self._curr_idx = 0

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
        block_idx = self._block_idxs[self._curr_idx]

        # Find the correct chunk and block within that chunk
        prev_blocks = 0
        for chunk_id, n_blocks in enumerate(self._n_blocks_list):
            if prev_blocks + n_blocks > block_idx:
                break
            prev_blocks += n_blocks

        local_block_idx = block_idx - prev_blocks
        buffer = self._buffers[chunk_id]
        elem_id = local_block_idx * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id

        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
        self._curr_idx += 1

        return torch.from_numpy(arr.astype(np.int64))


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)
