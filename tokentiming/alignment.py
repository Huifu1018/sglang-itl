"""Dynamic Token Warping for TokenTiming-style token alignment."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Callable, Sequence


DistanceFn = Callable[[str, str], float]


@dataclass(frozen=True)
class AlignmentStep:
    """One DTW path step mapping a draft token to a target token."""

    draft_index: int
    target_index: int
    cost: float


@dataclass(frozen=True)
class AlignmentResult:
    """DTW alignment plus convenient reverse indexes."""

    path: tuple[AlignmentStep, ...]
    total_cost: float
    target_to_draft: tuple[tuple[int, ...], ...]
    draft_to_target: tuple[tuple[int, ...], ...]


def levenshtein_distance(left: str, right: str) -> int:
    """Return character-level edit distance."""

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def dynamic_token_warping(
    draft_tokens: Sequence[str],
    target_tokens: Sequence[str],
    *,
    window: int | None = None,
    distance_fn: DistanceFn = levenshtein_distance,
) -> AlignmentResult:
    """Align draft tokens to retokenized target tokens with DTW.

    The paper constrains the DTW search with a Sakoe-Chiba style window. This
    implementation expands a too-small window to ``abs(m - n)`` so valid paths
    are not accidentally removed when the tokenizers produce different counts.
    """

    draft_tokens = tuple(draft_tokens)
    target_tokens = tuple(target_tokens)
    draft_count = len(draft_tokens)
    target_count = len(target_tokens)

    if draft_count == 0 and target_count == 0:
        return AlignmentResult(path=(), total_cost=0.0, target_to_draft=(), draft_to_target=())
    if draft_count == 0 or target_count == 0:
        raise ValueError("DTW alignment requires both token sequences to be non-empty.")

    if window is None:
        effective_window = max(draft_count, target_count)
    else:
        if window < 0:
            raise ValueError("window must be non-negative.")
        effective_window = max(window, abs(draft_count - target_count))

    dp = [[inf] * (target_count + 1) for _ in range(draft_count + 1)]
    back: list[list[tuple[int, int] | None]] = [
        [None] * (target_count + 1) for _ in range(draft_count + 1)
    ]
    dp[0][0] = 0.0

    for i in range(1, draft_count + 1):
        j_start = max(1, i - effective_window)
        j_end = min(target_count, i + effective_window)
        for j in range(j_start, j_end + 1):
            step_cost = float(distance_fn(draft_tokens[i - 1], target_tokens[j - 1]))
            candidates = (
                (dp[i - 1][j - 1], i - 1, j - 1),
                (dp[i - 1][j], i - 1, j),
                (dp[i][j - 1], i, j - 1),
            )
            best_cost, prev_i, prev_j = min(candidates, key=lambda item: item[0])
            if best_cost == inf:
                continue
            dp[i][j] = best_cost + step_cost
            back[i][j] = (prev_i, prev_j)

    if dp[draft_count][target_count] == inf:
        raise ValueError("No DTW path found for the supplied window.")

    reversed_path: list[AlignmentStep] = []
    i = draft_count
    j = target_count
    while i > 0 or j > 0:
        previous_cell = back[i][j]
        if previous_cell is None:
            raise ValueError("DTW backtracking failed; alignment is incomplete.")
        cost = float(distance_fn(draft_tokens[i - 1], target_tokens[j - 1]))
        reversed_path.append(AlignmentStep(draft_index=i - 1, target_index=j - 1, cost=cost))
        i, j = previous_cell

    path = tuple(reversed(reversed_path))
    target_to_draft_sets = [set() for _ in range(target_count)]
    draft_to_target_sets = [set() for _ in range(draft_count)]
    for step in path:
        target_to_draft_sets[step.target_index].add(step.draft_index)
        draft_to_target_sets[step.draft_index].add(step.target_index)

    return AlignmentResult(
        path=path,
        total_cost=float(dp[draft_count][target_count]),
        target_to_draft=tuple(tuple(sorted(indices)) for indices in target_to_draft_sets),
        draft_to_target=tuple(tuple(sorted(indices)) for indices in draft_to_target_sets),
    )


def token_strings_from_tokenizer(tokenizer: object, token_ids: Sequence[int]) -> tuple[str, ...]:
    """Decode each token id into the string unit used by DTW alignment."""

    return tuple(
        tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        for token_id in token_ids
    )
