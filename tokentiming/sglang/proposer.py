"""HF draft proposer used by the SGLang Token-ITL worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tokentiming.alignment import dynamic_token_warping

from .config import TokenITLSGLangConfig


@dataclass(frozen=True)
class DraftProposal:
    draft_token_ids: tuple[int, ...]
    proxy_target_token_ids: tuple[int, ...]
    alignment_cost: float | None


class HFDraftProposer:
    """Generate draft text and retokenize it into target-vocabulary proxies."""

    def __init__(
        self,
        *,
        draft_model_path: str,
        target_tokenizer: object,
        config: TokenITLSGLangConfig,
        trust_remote_code: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.target_tokenizer = target_tokenizer
        self.draft_tokenizer = AutoTokenizer.from_pretrained(
            draft_model_path,
            trust_remote_code=trust_remote_code,
        )

        model_kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
        if config.draft_dtype != "auto":
            model_kwargs["torch_dtype"] = _torch_dtype(torch, config.draft_dtype)
        else:
            model_kwargs["torch_dtype"] = "auto"
        if config.draft_device_map:
            model_kwargs["device_map"] = config.draft_device_map

        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path,
            **model_kwargs,
        )
        if not config.draft_device_map and config.draft_device:
            self.draft_model.to(config.draft_device)
        self.draft_model.eval()

    def propose(self, current_text: str, *, max_proxy_tokens: int) -> DraftProposal:
        """Return target-tokenizer proxy ids for a short draft continuation."""

        import torch

        if max_proxy_tokens <= 0:
            return DraftProposal((), (), None)

        encoded = self.draft_tokenizer(
            current_text,
            return_tensors="pt",
            add_special_tokens=self.config.add_special_tokens,
        )
        input_ids = encoded["input_ids"].to(self._input_device())
        attention_mask = torch.ones_like(input_ids)

        max_draft_tokens = self.config.max_draft_tokens
        if max_draft_tokens is None:
            max_draft_tokens = max(max_proxy_tokens * 4, max_proxy_tokens + 4)

        draft_ids: list[int] = []
        proxy_ids: list[int] = []
        past_key_values = None
        next_input = input_ids
        context = input_ids

        with torch.inference_mode():
            for _ in range(max_draft_tokens):
                outputs = self.draft_model(
                    input_ids=next_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                next_token = int(torch.argmax(logits, dim=-1)[0])
                draft_ids.append(next_token)

                draft_text = self._decode(self.draft_tokenizer, draft_ids)
                proxy_ids = self._encode(self.target_tokenizer, draft_text)
                if len(proxy_ids) >= max_proxy_tokens:
                    break

                next_tensor = torch.tensor(
                    [[next_token]],
                    dtype=context.dtype,
                    device=context.device,
                )
                context = torch.cat([context, next_tensor], dim=1)
                attention_mask = torch.ones_like(context)
                past_key_values = getattr(outputs, "past_key_values", None)
                next_input = next_tensor if past_key_values is not None else context

                eos_token_id = getattr(self.draft_tokenizer, "eos_token_id", None)
                if eos_token_id is not None and next_token == int(eos_token_id):
                    break

        proxy_ids = proxy_ids[:max_proxy_tokens]
        alignment_cost = self._alignment_cost(draft_ids, proxy_ids)
        return DraftProposal(
            draft_token_ids=tuple(draft_ids),
            proxy_target_token_ids=tuple(int(token_id) for token_id in proxy_ids),
            alignment_cost=alignment_cost,
        )

    def _input_device(self):
        import torch

        if self.config.draft_device and self.config.draft_device != "auto":
            return torch.device(self.config.draft_device)
        try:
            return next(self.draft_model.parameters()).device
        except StopIteration:
            return torch.device("cuda")

    def _alignment_cost(
        self,
        draft_ids: Sequence[int],
        proxy_ids: Sequence[int],
    ) -> float | None:
        if not draft_ids or not proxy_ids:
            return None
        try:
            draft_strings = tuple(self._decode(self.draft_tokenizer, [token_id]) for token_id in draft_ids)
            proxy_strings = tuple(self._decode(self.target_tokenizer, [token_id]) for token_id in proxy_ids)
            alignment = dynamic_token_warping(
                draft_strings,
                proxy_strings,
                window=self.config.dtw_window,
            )
            return alignment.total_cost
        except Exception:
            return None

    @staticmethod
    def _encode(tokenizer: object, text: str) -> list[int]:
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            encoded = tokenizer(text, add_special_tokens=False)
            input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return list(input_ids)

    @staticmethod
    def _decode(tokenizer: object, token_ids: Sequence[int]) -> str:
        ids = [int(token_id) for token_id in token_ids]
        try:
            return tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode(ids)


def _torch_dtype(torch_module: object, dtype_name: str) -> object:
    normalized = dtype_name.strip().lower()
    aliases = {
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
    }
    attr = aliases.get(normalized, normalized)
    if not hasattr(torch_module, attr):
        raise ValueError(f"Unsupported TOKEN_ITL_DRAFT_DTYPE: {dtype_name}")
    return getattr(torch_module, attr)
