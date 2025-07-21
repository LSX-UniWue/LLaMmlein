import json
import sys
from multiprocessing import Process
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.fa3_dataset as fa3_dataset
from lit_gpt import Tokenizer


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    max_seq_len: int,
    filenames_subset: List[str] = None,
    process_id: int = 0,
) -> None:
    destination_path = Path(destination_path)
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset

    print(filenames)
    if not filenames:
        raise RuntimeError(f"No files matching found at {source_path}. \n" "Make sure you download the data...")

    builder = fa3_dataset.FA3DatasetBuilder(
        outdir=destination_path,
        prefix=f"train_redpajama_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        max_seq_len=max_seq_len,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    validation_split = 0.01
    validation_builder = fa3_dataset.FA3DatasetBuilder(
        outdir=destination_path,
        prefix=f"valid_redpajama_{process_id}",
        chunk_size=chunk_size,
        max_seq_len=max_seq_len,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    for filepath in filenames:
        with open(filepath, encoding="utf-8") as f:
            print(f"Processing {filepath}")
            counter = 0
            for row in tqdm(f):
                data = json.loads(row)
                text_ids = tokenizer.encode(data["text"])

                if counter % int(1 / validation_split) == 0:
                    validation_builder.add_array(np.array(text_ids, dtype=validation_builder.dtype), data["id"])
                else:
                    builder.add_array(np.array(text_ids, dtype=builder.dtype), data["id"])

                # builder.add_array(np.array(text_ids, dtype=builder.dtype), data["id"]) # FIXME why was this here?
                counter += 1
                if counter % 10000 == 0:
                    print(f"Processed {counter} rows")

    builder.write_reminder()
    validation_builder.write_reminder()


def prepare(
    source_path: str,
    tokenizer_path: Path = Path("LLaMmlein_tok"),
    destination_path: Path = Path("data/preprocessed_fa3"),
    # chunk_size: int = 2049 * 1024 * 256,
    chunk_size: int = 10240,
    max_seq_len: int = 2048,
    num: int = 1,
) -> None:
    import time

    start_time = time.time()

    p = Process(
        target=prepare_full,
        args=(Path(source_path), tokenizer_path, destination_path, chunk_size, max_seq_len, [source_path], num),
    )
    p.start()
    p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset with specified path and number of processes.")
    parser.add_argument("--path", type=str, required=False, default="data/test.jsonl")
    parser.add_argument("--num", type=str, default="0")
    parser.add_argument("--destination_path", type=str, default="data/preprocessed_fa3")

    args = parser.parse_args()

    prepare(destination_path=args.destination_path, source_path=args.path, num=args.num)
