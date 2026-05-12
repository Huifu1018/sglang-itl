"""Configuration helpers for the SGLang Token-ITL plugin."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when set.")
    return parsed


@dataclass(frozen=True)
class TokenITLSGLangConfig:
    """Runtime knobs read by the Token-ITL SGLang worker.

    SGLang does not currently expose plugin-owned CLI flags through the public
    parser, so integration-specific knobs are environment variables.
    """

    draft_device: str | None = None
    draft_device_map: str | None = None
    draft_dtype: str = "auto"
    dtw_window: int | None = 8
    max_draft_tokens: int | None = None
    add_special_tokens: bool = False
    disable_cuda_graph: bool = True

    @classmethod
    def from_env(cls, *, default_draft_device: str | None = None) -> "TokenITLSGLangConfig":
        return cls(
            draft_device=os.getenv("TOKEN_ITL_DRAFT_DEVICE", default_draft_device),
            draft_device_map=os.getenv("TOKEN_ITL_DRAFT_DEVICE_MAP") or None,
            draft_dtype=os.getenv("TOKEN_ITL_DRAFT_DTYPE", "auto"),
            dtw_window=_env_int("TOKEN_ITL_DTW_WINDOW", 8),
            max_draft_tokens=_env_int("TOKEN_ITL_MAX_DRAFT_TOKENS", None),
            add_special_tokens=_env_bool("TOKEN_ITL_ADD_SPECIAL_TOKENS", False),
            disable_cuda_graph=_env_bool("TOKEN_ITL_DISABLE_CUDA_GRAPH", True),
        )
