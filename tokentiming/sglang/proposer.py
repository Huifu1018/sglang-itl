"""HF draft proposer used by the SGLang Token-ITL worker."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

from tokentiming.alignment import dynamic_token_warping

from .config import TokenITLSGLangConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DraftProposal:
    draft_token_ids: tuple[int, ...]
    proxy_target_token_ids: tuple[int, ...]
    alignment_cost: float | None
    cache_event: str
    draft_context_tokens: int


@dataclass
class DraftRequestState:
    rid: str
    text: str
    input_ids: tuple[int, ...]
    past_key_values: object
    next_token_logits: object


@dataclass
class DraftProposerStats:
    proposals: int = 0
    proposed_proxy_tokens: int = 0
    proposed_draft_tokens: int = 0
    cache_hits: int = 0
    cache_extensions: int = 0
    cache_rebuilds: int = 0
    cache_evictions: int = 0
    empty_proposals: int = 0
    failed_proposals: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "proposals": self.proposals,
            "proposed_proxy_tokens": self.proposed_proxy_tokens,
            "proposed_draft_tokens": self.proposed_draft_tokens,
            "cache_hits": self.cache_hits,
            "cache_extensions": self.cache_extensions,
            "cache_rebuilds": self.cache_rebuilds,
            "cache_evictions": self.cache_evictions,
            "empty_proposals": self.empty_proposals,
            "failed_proposals": self.failed_proposals,
        }


class HFDraftProposer:
    """Generate draft text and retokenize it into target-vocabulary proxies.

    The proposer keeps an HF ``past_key_values`` cache per SGLang request id.
    Cache reuse is conservative: it only extends a cached draft context when the
    newly encoded draft-token context has the cached token ids as an exact
    prefix. If tokenizer boundary effects make that unsafe, it rebuilds.
    """

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

        self._states: OrderedDict[str, DraftRequestState] = OrderedDict()
        self.stats = DraftProposerStats()

    def propose(
        self,
        rid: str,
        current_text: str,
        *,
        max_proxy_tokens: int,
    ) -> DraftProposal:
        """Return target-tokenizer proxy ids for a short draft continuation."""

        import torch

        self.stats.proposals += 1
        if max_proxy_tokens <= 0:
            self.stats.empty_proposals += 1
            return DraftProposal((), (), None, "disabled", 0)

        try:
            state, cache_event = self._ensure_state(rid, current_text)
            max_draft_tokens = self.config.max_draft_tokens
            if max_draft_tokens is None:
                max_draft_tokens = max(max_proxy_tokens * 4, max_proxy_tokens + 4)

            draft_ids: list[int] = []
            proxy_ids: list[int] = []
            generation_ids = list(state.input_ids)
            generation_past = state.past_key_values
            logits = state.next_token_logits
            context_len = len(state.input_ids)

            with torch.inference_mode():
                for _ in range(max_draft_tokens):
                    next_token = int(torch.argmax(logits, dim=-1)[0])
                    draft_ids.append(next_token)

                    draft_text = self._decode(self.draft_tokenizer, draft_ids)
                    proxy_ids = self._encode(self.target_tokenizer, draft_text)
                    if len(proxy_ids) >= max_proxy_tokens:
                        break

                    generation_ids.append(next_token)
                    context_len += 1
                    if generation_past is None:
                        next_tensor = torch.tensor(
                            [generation_ids],
                            dtype=torch.long,
                            device=self._input_device(),
                        )
                        attention_mask = torch.ones_like(next_tensor)
                    else:
                        next_tensor = torch.tensor(
                            [[next_token]],
                            dtype=torch.long,
                            device=self._input_device(),
                        )
                        attention_mask = torch.ones(
                            (1, context_len),
                            dtype=torch.long,
                            device=next_tensor.device,
                        )
                    outputs = self.draft_model(
                        input_ids=next_tensor,
                        attention_mask=attention_mask,
                        past_key_values=generation_past,
                        use_cache=True,
                    )
                    generation_past = getattr(outputs, "past_key_values", None)
                    logits = outputs.logits[:, -1, :]

                    eos_token_id = getattr(self.draft_tokenizer, "eos_token_id", None)
                    if eos_token_id is not None and next_token == int(eos_token_id):
                        break

            proxy_ids = proxy_ids[:max_proxy_tokens]
            if not proxy_ids:
                self.stats.empty_proposals += 1
            self.stats.proposed_proxy_tokens += len(proxy_ids)
            self.stats.proposed_draft_tokens += len(draft_ids)
            alignment_cost = self._alignment_cost(draft_ids, proxy_ids)
            return DraftProposal(
                draft_token_ids=tuple(draft_ids),
                proxy_target_token_ids=tuple(int(token_id) for token_id in proxy_ids),
                alignment_cost=alignment_cost,
                cache_event=cache_event,
                draft_context_tokens=len(state.input_ids),
            )
        except Exception:
            self.stats.failed_proposals += 1
            raise

    def evict(self, rids: Sequence[str]) -> None:
        for rid in rids:
            if self._states.pop(str(rid), None) is not None:
                self.stats.cache_evictions += 1

    def clear(self) -> None:
        evicted = len(self._states)
        self._states.clear()
        self.stats.cache_evictions += evicted

    def cache_size(self) -> int:
        return len(self._states)

    def _ensure_state(self, rid: str, current_text: str) -> tuple[DraftRequestState, str]:
        import torch

        rid = str(rid)
        context_ids = tuple(self._context_ids(current_text))
        if not context_ids:
            raise ValueError("draft context must contain at least one token.")

        cached = self._states.get(rid) if self.config.enable_draft_cache else None
        if cached is not None and cached.input_ids == context_ids:
            self._states.move_to_end(rid)
            self.stats.cache_hits += 1
            return cached, "hit"

        if (
            cached is not None
            and cached.past_key_values is not None
            and len(context_ids) > len(cached.input_ids)
            and context_ids[: len(cached.input_ids)] == cached.input_ids
        ):
            suffix = context_ids[len(cached.input_ids) :]
            suffix_tensor = torch.tensor(
                [list(suffix)],
                dtype=torch.long,
                device=self._input_device(),
            )
            attention_mask = torch.ones(
                (1, len(context_ids)),
                dtype=torch.long,
                device=suffix_tensor.device,
            )
            with torch.inference_mode():
                outputs = self.draft_model(
                    input_ids=suffix_tensor,
                    attention_mask=attention_mask,
                    past_key_values=cached.past_key_values,
                    use_cache=True,
                )
            state = DraftRequestState(
                rid=rid,
                text=current_text,
                input_ids=context_ids,
                past_key_values=getattr(outputs, "past_key_values", None),
                next_token_logits=outputs.logits[:, -1, :],
            )
            self._store_state(state)
            self.stats.cache_extensions += 1
            return state, "extend"

        input_tensor = torch.tensor(
            [list(context_ids)],
            dtype=torch.long,
            device=self._input_device(),
        )
        attention_mask = torch.ones_like(input_tensor)
        with torch.inference_mode():
            outputs = self.draft_model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=True,
            )
        state = DraftRequestState(
            rid=rid,
            text=current_text,
            input_ids=context_ids,
            past_key_values=getattr(outputs, "past_key_values", None),
            next_token_logits=outputs.logits[:, -1, :],
        )
        self._store_state(state)
        self.stats.cache_rebuilds += 1
        return state, "rebuild"

    def _store_state(self, state: DraftRequestState) -> None:
        if not self.config.enable_draft_cache:
            return
        self._states[state.rid] = state
        self._states.move_to_end(state.rid)
        while len(self._states) > self.config.max_cached_requests:
            self._states.popitem(last=False)
            self.stats.cache_evictions += 1

    def _context_ids(self, text: str) -> list[int]:
        token_ids = self._encode(
            self.draft_tokenizer,
            text,
            add_special_tokens=self.config.add_special_tokens,
        )
        max_context_tokens = self.config.max_context_tokens
        if max_context_tokens is not None and len(token_ids) > max_context_tokens:
            token_ids = token_ids[-max_context_tokens:]
        return token_ids

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
            draft_strings = tuple(
                self._decode(self.draft_tokenizer, [token_id])
                for token_id in draft_ids
            )
            proxy_strings = tuple(
                self._decode(self.target_tokenizer, [token_id])
                for token_id in proxy_ids
            )
            alignment = dynamic_token_warping(
                draft_strings,
                proxy_strings,
                window=self.config.dtw_window,
            )
            return alignment.total_cost
        except Exception:
            return None

    @staticmethod
    def _encode(
        tokenizer: object,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        try:
            return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
        except TypeError:
            encoded = tokenizer(text, add_special_tokens=add_special_tokens)
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
