"""Configuration helpers for the SGLang TOKEN_ITL plugin."""

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


def _env_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in {"0", "false", "off", "none"}:
        return None
    parsed = float(value)
    if parsed == 0:
        return None
    if parsed < 0:
        raise ValueError(f"{name} must be positive when set.")
    return parsed


@dataclass(frozen=True)
class TokenITLSGLangConfig:
    """Runtime knobs read by the SGLang TOKEN_ITL worker.

    SGLang does not currently expose plugin-owned CLI flags through the public
    parser, so integration-specific knobs are environment variables.
    """

    draft_device: str | None = None
    draft_device_map: str | None = None
    draft_dtype: str = "auto"
    dtw_window: int | None = 8
    max_draft_tokens: int | None = None
    max_context_tokens: int | None = None
    max_cached_requests: int = 256
    add_special_tokens: bool = False
    disable_cuda_graph: bool = True
    enable_draft_cache: bool = True
    clone_draft_cache: bool = True
    metrics_log_interval: float | None = 60.0

    @classmethod
    def from_env(cls, *, default_draft_device: str | None = None) -> "TokenITLSGLangConfig":
        return cls(
            draft_device=os.getenv("TOKEN_ITL_DRAFT_DEVICE", default_draft_device),
            draft_device_map=os.getenv("TOKEN_ITL_DRAFT_DEVICE_MAP") or None,
            draft_dtype=os.getenv("TOKEN_ITL_DRAFT_DTYPE", "auto"),
            dtw_window=_env_int("TOKEN_ITL_DTW_WINDOW", 8),
            max_draft_tokens=_env_int("TOKEN_ITL_MAX_DRAFT_TOKENS", None),
            max_context_tokens=_env_int("TOKEN_ITL_MAX_CONTEXT_TOKENS", None),
            max_cached_requests=_env_int("TOKEN_ITL_MAX_CACHED_REQUESTS", 256) or 256,
            add_special_tokens=_env_bool("TOKEN_ITL_ADD_SPECIAL_TOKENS", False),
            disable_cuda_graph=_env_bool("TOKEN_ITL_DISABLE_CUDA_GRAPH", True),
            enable_draft_cache=_env_bool("TOKEN_ITL_ENABLE_DRAFT_CACHE", True),
            clone_draft_cache=_env_bool("TOKEN_ITL_CLONE_DRAFT_CACHE", True),
            metrics_log_interval=_env_float("TOKEN_ITL_METRICS_LOG_INTERVAL", 60.0),
        )
