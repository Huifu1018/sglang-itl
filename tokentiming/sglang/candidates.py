"""Candidate-row helpers for the SGLang TOKEN_ITL worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CandidateRows:
    rows: tuple[tuple[int, ...], ...]
    draft_token_num: int
    proposed_proxy_tokens: int


def build_linear_candidate_rows(
    roots: Sequence[int],
    proxy_rows: Sequence[Sequence[int]],
    *,
    max_draft_token_num: int,
) -> CandidateRows:
    """Build equal-width linear verify rows without padding fake candidates.

    SGLang's target verifier expects one width per batch. To avoid verifying
    artificial pad/eos tokens, the width is the shortest real row in the batch.
    Width 1 is valid and means target-only decode for that batch.
    """

    if max_draft_token_num <= 0:
        raise ValueError("max_draft_token_num must be positive.")
    if len(roots) != len(proxy_rows):
        raise ValueError("roots and proxy_rows must have the same length.")

    raw_rows: list[tuple[int, ...]] = []
    proposed_proxy_tokens = 0
    max_proxy_tokens = max(0, max_draft_token_num - 1)
    for root, proxies in zip(roots, proxy_rows):
        clipped = tuple(int(token_id) for token_id in proxies[:max_proxy_tokens])
        proposed_proxy_tokens += len(clipped)
        raw_rows.append((int(root), *clipped))

    if not raw_rows:
        return CandidateRows(rows=(), draft_token_num=1, proposed_proxy_tokens=0)

    draft_token_num = max(1, min(len(row) for row in raw_rows))
    draft_token_num = min(draft_token_num, max_draft_token_num)
    rows = tuple(row[:draft_token_num] for row in raw_rows)
    return CandidateRows(
        rows=rows,
        draft_token_num=draft_token_num,
        proposed_proxy_tokens=proposed_proxy_tokens,
    )
