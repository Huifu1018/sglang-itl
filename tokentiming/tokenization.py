"""Tokenizer adapter utilities used by TokenTiming decoders."""

from __future__ import annotations

from typing import Sequence


class TokenizerAdapter:
    """Small compatibility layer over Hugging Face tokenizers."""

    def __init__(self, tokenizer: object, *, add_special_tokens: bool = False) -> None:
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def encode_tensor(self, text: str, *, device: str | None = None):
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens,
        )
        input_ids = encoded["input_ids"]
        if device is not None:
            input_ids = input_ids.to(device)
        return input_ids

    def encode_ids(self, text: str) -> list[int]:
        return list(
            self.tokenizer.encode(
                text,
                add_special_tokens=self.add_special_tokens,
            )
        )

    def decode_ids(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def token_strings(self, token_ids: Sequence[int]) -> tuple[str, ...]:
        return tuple(self.decode_ids([int(token_id)]) for token_id in token_ids)
