"""Result and trace structures for TokenTiming generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .prob_mapping import ProposalProbability


@dataclass(frozen=True)
class VerificationTrace:
    """Debug trace for one draft block."""

    step_index: int
    draft_token_ids: tuple[int, ...]
    proxy_target_token_ids: tuple[int, ...]
    proposal_probabilities: tuple[ProposalProbability, ...]
    alignment_cost: float
    accepted_tokens: int
    rejected: bool
    replacement_token_id: int | None


@dataclass
class GenerationStats:
    """Counters that are useful when deciding whether a draft pair is worth using."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    draft_forwards: int = 0
    target_forwards: int = 0
    proposed_proxy_tokens: int = 0
    accepted_proxy_tokens: int = 0
    accepted_blocks: int = 0
    rejected_blocks: int = 0
    elapsed_seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_proxy_tokens == 0:
            return 0.0
        return self.accepted_proxy_tokens / self.proposed_proxy_tokens

    @property
    def tokens_per_target_forward(self) -> float:
        if self.target_forwards == 0:
            return 0.0
        return self.generated_tokens / self.target_forwards


@dataclass(frozen=True)
class GenerationResult:
    """Full-text generation result plus traces and aggregate stats."""

    text: str
    generated_text: str
    token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    stats: GenerationStats
    traces: tuple[VerificationTrace, ...] = field(default_factory=tuple)
