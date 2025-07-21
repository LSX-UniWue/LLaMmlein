# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import io
import json
from pathlib import Path
from typing import Optional

import torch
from nltk.tokenize.punkt import PunktSentenceTokenizer
from torch.utils.data import IterableDataset, get_worker_info

from lit_gpt import Tokenizer
import re

from .packed_dataset import CombinedDataset, CombinedDatasetIterator


class JsonlDataset(IterableDataset):
    def __init__(
        self,
        filenames,
        block_size,
        tokenizer_path: Path,
        padding_id,
        seed=12345,
        wrap=False,
        num_processes=1,
        process_rank=0,
        resume: Optional[Path] = None,
    ):
        self._filenames = filenames
        self._block_size = block_size
        self._seed = seed
        self._tokenizer_path = tokenizer_path
        self._padding_id = padding_id

        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank
        self._resume = resume

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        assert worker_id == 0, "Only one worker is supported (?)"
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        return JsonlDatasetIterator(
            filenames=filenames,
            block_size=self._block_size,
            seed=self._seed,
            tokenizer_path=self._tokenizer_path,
            padding_id=self._padding_id,
            wrap=self._wrap,
            process_rank=self._process_rank,
            resume=self._resume,
        )


class JsonlDatasetIterator:
    def __init__(
        self,
        filenames,
        block_size,
        seed,
        wrap,
        tokenizer_path,
        padding_id,
        process_rank,
        resume: Optional[Path],
    ):
        self._seed = seed
        self._block_idxs = None
        self._tokenizer = Tokenizer(tokenizer_path)
        self._padding_id = padding_id
        self._process_rank = process_rank
        self._resume = resume

        self._file_idx = 0
        if self._resume is not None:
            # log_file_path = resume.with_name(resume.stem.replace("-ckpt", "-{rank}.log"))
            self._resume = Path(str(self._resume).format(rank=self._process_rank))
            assert self._resume.exists(), self._resume

            print(f"Resuming from {self._resume}")
            # load last line from file as jsonl, get id, and resume from there
            with open(self._resume, "r") as f:
                for line in f:  # yes, it's bad, but it's only 7 MB
                    pass
                resume_meta_data = json.loads(line)
                self._resume_id = resume_meta_data["data_id"][-1]
                self._file_idx = resume_meta_data["file_id"][-1] if "file_id" in resume_meta_data else 0

        self._wrap = wrap

        self._filenames = filenames

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None
        self._n_blocks_list = []

        self._block_idxs = []
        self._curr_idx = 0
        self._tmp_files = []
        self._sent_tokenizer = PunktSentenceTokenizer()

        self._open_file(self._filenames[self._file_idx])
        if self._resume is not None:
            self._continue_from_last()

    def _continue_from_last(self):
        if self._resume is None:
            return
        print(
            f"Start resuming {self._process_rank} from {self._resume_id} from within file {self._file_idx} {self._filenames[self._file_idx]}"
        )
        # Keep calling next until we find the last processed ID
        while True:
            item = self.__next__()
            if item["data_id"] == self._resume_id:
                print(
                    f"Resuming from {self._resume_id} on dataset {self._filenames[self._file_idx]} for process {self._process_rank}"
                )
                break

    def __iter__(self):
        return self

    def _open_file(self, filename):
        try:
            self._current_file.close()
        except:
            pass
        self._current_file = open(
            filename,
            mode="r",
            buffering=io.DEFAULT_BUFFER_SIZE * 1000,
        )

    def _next_datapoint(self) -> dict[str, str]:
        try:
            line = self._current_file.readline()
            if line == "":
                raise EOFError
        except:
            line = None
            self._file_idx += 1
            if self._file_idx >= len(self._filenames):
                self._file_idx = 0
            self._open_file(self._filenames[self._file_idx])

        if not line:
            return self._next_datapoint()

        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {line.strip()} (Error: {e})")
            return self._next_datapoint()  # Skip the bad line and continue

    def _semantically_next_sentence_split_datapoint(self):
        # slooooow as frick, but dunno why
        while True:
            datapoint = self._next_datapoint()
            tokenized = self._tokenizer.encode(datapoint["text"])
            if len(tokenized) <= self._block_size + 1:
                datapoint["tokenized"] = tokenized
                yield datapoint
            else:
                # continue
                # repeatedly retokenize until it is just below the limit
                sentences = self._sent_tokenizer.tokenize(datapoint["text"])
                # Try all possible combinations of sentences that fit within block_size
                for i in range(len(sentences)):
                    candidate_texts = []
                    current_combination = []
                    for j in range(i, len(sentences)):
                        current_combination.append(sentences[j])
                        candidate_texts.append(" ".join(current_combination))

                    if not candidate_texts:
                        continue

                    # Batch encode all possible combinations
                    tokenized_batch = self._tokenizer.batch_encode(candidate_texts)

                    # Find the longest combination that fits
                    valid_idx = -1
                    for idx, tokens in enumerate(tokenized_batch):
                        if len(tokens) <= self._block_size + 1:
                            valid_idx = idx
                        else:
                            break

                    if valid_idx >= 0:
                        split_datapoint = {
                            "id": datapoint["id"],
                            "tokenized": tokenized_batch[valid_idx],
                        }
                        yield split_datapoint
                        # Skip ahead by the number of sentences we used
                        i += valid_idx
                    else:
                        # If no valid combination found, skip this sentence
                        continue

    def _next_sentence_split_datapoint(self):
        while True:
            datapoint = self._next_datapoint()
            tokenized = self._tokenizer.encode(datapoint["text"], bos=False, eos=True)
            if len(tokenized) <= self._block_size + 1:
                datapoint["tokenized"] = tokenized
                yield datapoint
            else:
                # chunk it into equal datapoints up to self._block_size + 1
                tokenized_chunks = [
                    tokenized[i : i + self._block_size + 1] for i in range(0, len(tokenized), self._block_size + 1)
                ]
                for chunk in tokenized_chunks:
                    split_datapoint = {
                        "id": datapoint["id"],
                        "tokenized": chunk,
                    }
                    yield split_datapoint

    def __next__(self):
        if not hasattr(self, '_sentence_split_iterator'):
            self._sentence_split_iterator = self._next_sentence_split_datapoint()

        datapoint = next(self._sentence_split_iterator)
        padding_tensor = torch.tensor([self._padding_id] * ((self._block_size - len(datapoint["tokenized"])) + 1))
        datapoint["tokenized"] = torch.cat((datapoint["tokenized"], padding_tensor)).long()
        datapoint = {k: datapoint[k] for k in ["id", "tokenized"]}

        return {
            "input_ids": datapoint["tokenized"],
            "data_id": datapoint["id"],
            "file_id": self._file_idx,
            "process_rank": self._process_rank,
        }

