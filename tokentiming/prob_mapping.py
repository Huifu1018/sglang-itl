"""Map draft-token probabilities onto target proxy tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .alignment import AlignmentResult


@dataclass(frozen=True)
class ProposalProbability:
    """Proposal probability assigned to one target proxy token."""

    target_index: int
    target_token_id: int
    draft_indices: tuple[int, ...]
    source_draft_index: int
    source_draft_token_id: int
    probability: float


def map_top1_draft_probabilities(
    draft_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    alignment: AlignmentResult,
    draft_token_probabilities: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> tuple[ProposalProbability, ...]:
    """Project top-1 draft probabilities to target proxy tokens.

    TokenTiming aligns generated draft tokens against target proxy tokens after
    retokenization. For many-to-one alignment, the paper uses the terminal
    draft token as the source probability; for one-to-many, the same draft
    probability is reused for each target token it maps to.
    """

    draft_token_ids = tuple(int(token_id) for token_id in draft_token_ids)
    target_token_ids = tuple(int(token_id) for token_id in target_token_ids)
    probabilities = tuple(float(probability) for probability in draft_token_probabilities)

    if len(draft_token_ids) != len(probabilities):
        raise ValueError("draft_token_ids and draft_token_probabilities must have same length.")
    if len(target_token_ids) != len(alignment.target_to_draft):
        raise ValueError("target_token_ids length does not match alignment target length.")

    mapped: list[ProposalProbability] = []
    for target_index, draft_indices in enumerate(alignment.target_to_draft):
        if not draft_indices:
            raise ValueError(f"target token at index {target_index} has no draft alignment.")
        source_draft_index = max(draft_indices)
        probability = probabilities[source_draft_index]
        if epsilon:
            probability = min(1.0 - epsilon, max(epsilon, probability))
        mapped.append(
            ProposalProbability(
                target_index=target_index,
                target_token_id=target_token_ids[target_index],
                draft_indices=tuple(draft_indices),
                source_draft_index=source_draft_index,
                source_draft_token_id=draft_token_ids[source_draft_index],
                probability=probability,
            )
        )

    return tuple(mapped)


def acceptance_probability(
    target_probability: float,
    proposal_probability: float,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return speculative acceptance probability ``min(1, q(x) / p(x))``."""

    if target_probability < 0 or proposal_probability < 0:
        raise ValueError("probabilities must be non-negative.")
    proposal_probability = max(float(proposal_probability), epsilon)
    return min(1.0, float(target_probability) / proposal_probability)


def selected_token_probabilities_from_logits(
    logits: object,
    selected_token_ids: Sequence[int],
    *,
    temperature: float = 1.0,
) -> tuple[float, ...]:
    """Return p(selected token) from a ``[steps, vocab]`` PyTorch logits tensor."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    import torch

    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [steps, vocab].")
    selected = torch.as_tensor(tuple(int(token_id) for token_id in selected_token_ids), device=logits.device)
    if logits.shape[0] != selected.numel():
        raise ValueError("logits first dimension must match selected_token_ids length.")

    probabilities = torch.softmax(logits / temperature, dim=-1)
    step_indices = torch.arange(selected.numel(), device=logits.device)
    chosen = probabilities[step_indices, selected]
    return tuple(float(value) for value in chosen.detach().cpu())
