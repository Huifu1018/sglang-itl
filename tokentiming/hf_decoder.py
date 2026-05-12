"""Hugging Face TokenTiming greedy verifier.

This module implements a practical, deterministic TokenTiming integration:

1. Decode a short draft block with the draft model.
2. Decode draft token ids to text and retokenize that text with the target tokenizer.
3. Use Dynamic Token Warping to align draft tokens and target proxy tokens.
4. Verify all proxy target tokens with one batched target-model forward.

For greedy decoding, the returned text is identical to target-model greedy decoding.
The draft model only reduces the number of target forwards when enough proxy tokens
are accepted per block. Stochastic residual sampling is intentionally not exposed
here because an industrial lossless sampler needs a full target-vocabulary proposal
distribution, not only the selected draft-token probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

from .alignment import dynamic_token_warping
from .config import TokenTimingConfig
from .prob_mapping import (
    map_top1_draft_probabilities,
    selected_token_probabilities_from_logits,
)
from .result import GenerationResult, GenerationStats, VerificationTrace
from .tokenization import TokenizerAdapter


@dataclass(frozen=True)
class DraftBlock:
    token_ids: tuple[int, ...]
    logits: tuple[object, ...]
    forward_count: int


@dataclass(frozen=True)
class VerificationResult:
    token_ids: object
    accepted_tokens: int
    rejected: bool
    replacement_token_id: int | None


class TokenTimingGreedyDecoder:
    """Production-oriented greedy TokenTiming decoder for HF CausalLM models."""

    def __init__(
        self,
        target_model: object,
        draft_model: object,
        target_tokenizer: object,
        draft_tokenizer: object,
        config: TokenTimingConfig | None = None,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.target_tokens = TokenizerAdapter(
            target_tokenizer,
            add_special_tokens=(config.add_special_tokens if config else False),
        )
        self.draft_tokens = TokenizerAdapter(
            draft_tokenizer,
            add_special_tokens=(config.add_special_tokens if config else False),
        )
        self.config = config or TokenTimingConfig()
        self.config.validate()

        if hasattr(self.target_model, "eval"):
            self.target_model.eval()
        if hasattr(self.draft_model, "eval"):
            self.draft_model.eval()

    def generate(self, prompt: str) -> GenerationResult:
        """Generate text with target-greedy-equivalent verification."""

        import torch

        target_ids = self.target_tokens.encode_tensor(
            prompt,
            device=self.config.effective_target_device,
        )
        if target_ids.shape[1] == 0:
            raise ValueError("prompt must encode to at least one target token.")

        prompt_length = int(target_ids.shape[1])
        stats = GenerationStats(prompt_tokens=prompt_length)
        traces: list[VerificationTrace] = []
        started_at = perf_counter()

        with torch.inference_mode():
            while int(target_ids.shape[1]) - prompt_length < self.config.max_new_tokens:
                remaining = self.config.max_new_tokens - (int(target_ids.shape[1]) - prompt_length)
                current_text = self.target_tokens.decode_ids(target_ids[0].tolist())
                draft_context_ids = self.draft_tokens.encode_tensor(
                    current_text,
                    device=self.config.effective_draft_device,
                )

                draft_block = self._generate_draft_block(draft_context_ids)
                stats.draft_forwards += draft_block.forward_count
                if not draft_block.token_ids:
                    target_ids = self._append_target_greedy(target_ids)
                    stats.target_forwards += 1
                    if self._ends_with_eos(target_ids):
                        break
                    continue

                draft_text = self.draft_tokens.decode_ids(draft_block.token_ids)
                proxy_ids = self.target_tokens.encode_ids(draft_text)
                proxy_ids = self._limit_proxy_ids(proxy_ids, remaining)
                if not proxy_ids:
                    target_ids = self._append_target_greedy(target_ids)
                    stats.target_forwards += 1
                    if self._ends_with_eos(target_ids):
                        break
                    continue

                draft_strings = self.draft_tokens.token_strings(draft_block.token_ids)
                proxy_strings = self.target_tokens.token_strings(proxy_ids)
                alignment = dynamic_token_warping(
                    draft_strings,
                    proxy_strings,
                    window=self.config.dtw_window,
                )
                draft_probabilities = selected_token_probabilities_from_logits(
                    torch.stack(list(draft_block.logits)),
                    draft_block.token_ids,
                    temperature=self.config.temperature,
                )
                proposal_probabilities = map_top1_draft_probabilities(
                    draft_block.token_ids,
                    proxy_ids,
                    alignment,
                    draft_probabilities,
                )

                verified = self._verify_proxy_tokens(target_ids, proxy_ids)
                stats.target_forwards += 1
                stats.proposed_proxy_tokens += len(proxy_ids)
                stats.accepted_proxy_tokens += verified.accepted_tokens
                if verified.rejected:
                    stats.rejected_blocks += 1
                else:
                    stats.accepted_blocks += 1

                traces.append(
                    VerificationTrace(
                        step_index=len(traces),
                        draft_token_ids=draft_block.token_ids,
                        proxy_target_token_ids=tuple(proxy_ids),
                        proposal_probabilities=proposal_probabilities,
                        alignment_cost=alignment.total_cost,
                        accepted_tokens=verified.accepted_tokens,
                        rejected=verified.rejected,
                        replacement_token_id=verified.replacement_token_id,
                    )
                )

                target_ids = verified.token_ids
                if self._ends_with_eos(target_ids):
                    break

        elapsed = perf_counter() - started_at
        token_ids = tuple(int(token_id) for token_id in target_ids[0].tolist())
        generated_token_ids = token_ids[prompt_length:]
        text = self.target_tokens.decode_ids(token_ids)
        generated_text = self.target_tokens.decode_ids(generated_token_ids)
        stats.generated_tokens = len(generated_token_ids)
        stats.elapsed_seconds = elapsed

        return GenerationResult(
            text=text,
            generated_text=generated_text,
            token_ids=token_ids,
            generated_token_ids=generated_token_ids,
            stats=stats,
            traces=tuple(traces),
        )

    def _generate_draft_block(self, input_ids: object) -> DraftBlock:
        """Generate a greedy draft block, using KV cache when available."""

        import torch

        generated: list[int] = []
        logits_per_step: list[object] = []
        forward_count = 0
        context = input_ids
        attention_mask = torch.ones_like(context)
        past_key_values = None
        next_input = context
        use_incremental = self.config.use_cache

        for _ in range(self.config.num_draft_tokens):
            outputs = self._forward_model(
                self.draft_model,
                input_ids=next_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_incremental,
            )
            forward_count += 1
            logits = outputs.logits[:, -1, :]
            next_token = int(torch.argmax(logits, dim=-1)[0])
            generated.append(next_token)
            logits_per_step.append(logits[0].detach())

            next_tensor = torch.tensor([[next_token]], dtype=context.dtype, device=context.device)
            context = torch.cat([context, next_tensor], dim=1)
            attention_mask = torch.ones_like(context)

            if use_incremental and hasattr(outputs, "past_key_values"):
                past_key_values = outputs.past_key_values
                next_input = next_tensor
            else:
                past_key_values = None
                next_input = context

            if self.config.eos_token_id is not None and next_token == self.config.eos_token_id:
                break

        return DraftBlock(
            token_ids=tuple(generated),
            logits=tuple(logits_per_step),
            forward_count=forward_count,
        )

    def _verify_proxy_tokens(self, target_ids: object, proxy_ids: Sequence[int]) -> VerificationResult:
        """Verify a proxy block with one target forward."""

        import torch

        proxy_tensor = torch.tensor(
            [list(int(token_id) for token_id in proxy_ids)],
            dtype=target_ids.dtype,
            device=target_ids.device,
        )
        context_length = int(target_ids.shape[1])
        verify_input = torch.cat([target_ids, proxy_tensor], dim=1)
        outputs = self._forward_model(self.target_model, input_ids=verify_input)
        logits = outputs.logits[0]

        positions = torch.arange(
            context_length - 1,
            context_length + len(proxy_ids) - 1,
            device=logits.device,
        )
        predicted_ids = torch.argmax(logits.index_select(0, positions), dim=-1)
        proxy_tensor_flat = proxy_tensor[0].to(predicted_ids.device)
        matches = predicted_ids.eq(proxy_tensor_flat)

        if bool(matches.all()):
            accepted_ids = list(int(token_id) for token_id in proxy_ids)
            accepted_ids = self._truncate_after_eos(accepted_ids)
            if len(accepted_ids) == len(proxy_ids):
                return VerificationResult(
                    token_ids=verify_input,
                    accepted_tokens=len(proxy_ids),
                    rejected=False,
                    replacement_token_id=None,
                )
            appended = torch.tensor([accepted_ids], dtype=target_ids.dtype, device=target_ids.device)
            return VerificationResult(
                token_ids=torch.cat([target_ids, appended], dim=1),
                accepted_tokens=len(accepted_ids),
                rejected=False,
                replacement_token_id=None,
            )

        reject_index = int((~matches).nonzero(as_tuple=False)[0][0])
        replacement_token_id = int(predicted_ids[reject_index])
        appended_ids = list(int(token_id) for token_id in proxy_ids[:reject_index])
        appended_ids.append(replacement_token_id)
        appended_ids = self._truncate_after_eos(appended_ids)
        appended = torch.tensor([appended_ids], dtype=target_ids.dtype, device=target_ids.device)

        return VerificationResult(
            token_ids=torch.cat([target_ids, appended], dim=1),
            accepted_tokens=reject_index,
            rejected=True,
            replacement_token_id=replacement_token_id,
        )

    def _append_target_greedy(self, target_ids: object) -> object:
        import torch

        outputs = self._forward_model(self.target_model, input_ids=target_ids)
        next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1)[0])
        next_tensor = torch.tensor([[next_token]], dtype=target_ids.dtype, device=target_ids.device)
        return torch.cat([target_ids, next_tensor], dim=1)

    def _forward_model(self, model: object, **kwargs: object) -> object:
        """Call a HF model while tolerating models with narrower forward signatures."""

        try:
            return model(**{key: value for key, value in kwargs.items() if value is not None})
        except TypeError:
            fallback = {
                "input_ids": kwargs["input_ids"],
            }
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                fallback["attention_mask"] = kwargs["attention_mask"]
            return model(**fallback)

    def _limit_proxy_ids(self, proxy_ids: Sequence[int], remaining_tokens: int) -> list[int]:
        limited = list(int(token_id) for token_id in proxy_ids[:remaining_tokens])
        if self.config.max_proxy_tokens_per_step is not None:
            limited = limited[: self.config.max_proxy_tokens_per_step]
        return limited

    def _truncate_after_eos(self, token_ids: Sequence[int]) -> list[int]:
        token_ids = list(int(token_id) for token_id in token_ids)
        eos_token_id = self.config.eos_token_id
        if eos_token_id is None:
            return token_ids
        if eos_token_id not in token_ids:
            return token_ids
        eos_index = token_ids.index(eos_token_id)
        return token_ids[: eos_index + 1]

    def _ends_with_eos(self, token_ids: object) -> bool:
        return (
            self.config.eos_token_id is not None
            and int(token_ids[0, -1]) == int(self.config.eos_token_id)
        )
