import json
from pathlib import Path
from typing import Optional, Union

import torch


class Tokenizer:
    def __init__(self, checkpoint_dir: Path) -> None:
        # some checkpoints have both files, hugginface takes precedence => fasttok with batch enc
        # this is literally unusable for other people. turn around, don't go here.
        self.processor: Union["SentencePieceProcessor", "LlamaTokenizerFast", "AutoTokenizer"]
        if (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            try:
                from transformers import AutoTokenizer, LlamaTokenizerFast

                self.processor = AutoTokenizer.from_pretrained(str(checkpoint_dir))
                print(f"Using fast LlamaTokenizerFast")
            except:
                from tokenizers import Tokenizer as HFTokenizer

                print(":sadge:, slow tokenizer")
                self.processor = HFTokenizer.from_file(str(vocabulary_path))
            from sentencepiece import SentencePieceProcessor

            self.backup_processor = SentencePieceProcessor(model_file=str(checkpoint_dir / "tokenizer.model"))
            self.backend = "huggingface"
            with open(checkpoint_dir / "tokenizer_config.json") as fp:
                config = json.load(fp)
            bos_token = config.get("bos_token")
            self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
            self.eos_id = self.token_to_id(config["eos_token"])
        elif (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            from sentencepiece import SentencePieceProcessor

            self.processor = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.eos_id = self.processor.eos_id()
        else:
            raise NotImplementedError
        print(f"tokenizer: {self.processor=}, {type(self.processor)=}")

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=False)
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        raise RuntimeError

    def token_to_id(self, token: str) -> int:
        if self.backend == "huggingface":
            id_ = self.processor.convert_tokens_to_ids(token)
        elif self.backend == "sentencepiece":
            id_ = self.processor.piece_to_id(token)
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: bool = False,
        eos: bool = True,
        max_length: int = -1,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            tokens = self.processor.encode(string)
        elif self.backend == "sentencepiece":
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError
        if bos:
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError("This tokenizer does not defined a bos token")
            tokens = [bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def batch_encode(
        self,
        strings: list[str],
        device: Optional[torch.device] = None,
        bos: bool = False,
        eos: bool = True,
        max_length: int = -1,
    ) -> list[torch.Tensor]:
        if self.backend == "huggingface":
            batch_tokens = self.processor.batch_encode_plus(strings, add_special_tokens=False, return_tensors=None)
            batch_tokens = [tokens for tokens in batch_tokens["input_ids"]]
        elif self.backend == "sentencepiece":
            # no idea whether this actually exists
            batch_tokens = self.processor.encode(strings)
            batch_tokens = [tokens for tokens in batch_tokens]
        else:
            raise RuntimeError

        if bos:
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError("This tokenizer does not defined a bos token")
            batch_tokens = [[bos_id] + tokens for tokens in batch_tokens]
        if eos:
            eos_id = self.eos_id
            batch_tokens = [tokens + [eos_id] for tokens in batch_tokens]
        if max_length > 0:
            batch_tokens = [tokens[:max_length] for tokens in batch_tokens]
        return [torch.tensor(tokens, dtype=torch.int, device=device) for tokens in batch_tokens]

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens)
