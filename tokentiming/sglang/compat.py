"""Compatibility helpers for older SGLang speculative APIs."""

from __future__ import annotations

import os
from typing import Callable


LEGACY_PATCH_ENV = "TOKEN_ITL_LEGACY_NGRAM_PATCH"


def has_native_custom_spec_registry() -> bool:
    """Return whether SGLang exposes the out-of-tree spec registry API."""

    try:
        import sglang.srt.speculative.spec_registry  # noqa: F401
    except Exception:
        return False
    return True


def patch_legacy_ngram_worker() -> bool:
    """Patch SGLang 0.5.9/0.5.10 to route NGRAM to TOKEN_ITL on demand.

    SGLang 0.5.9 does not expose `sglang.srt.plugins` loading or a custom
    speculative algorithm registry. The least invasive fallback is to keep
    SGLang's built-in NGRAM algorithm name during argument parsing and replace
    only the worker factory when `TOKEN_ITL_LEGACY_NGRAM_PATCH=1`.
    """

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if getattr(SpeculativeAlgorithm, "_token_itl_legacy_patch", False):
        return True

    original_create_worker: Callable = SpeculativeAlgorithm.create_worker

    def create_worker(self, server_args):
        if (
            os.getenv(LEGACY_PATCH_ENV) == "1"
            and self == SpeculativeAlgorithm.NGRAM
        ):
            from .worker import TokenITLWorker

            return TokenITLWorker
        return original_create_worker(self, server_args)

    SpeculativeAlgorithm.create_worker = create_worker
    SpeculativeAlgorithm._token_itl_legacy_patch = True
    return True
