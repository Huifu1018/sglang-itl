"""Launch SGLang with TOKEN_ITL across native and legacy SGLang APIs."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from tokentiming.sglang.compat import (
    LEGACY_PATCH_ENV,
    has_native_custom_spec_registry,
    patch_legacy_ngram_worker,
)


TOKEN_ITL = "TOKEN_ITL"


def main() -> None:
    argv = list(sys.argv[1:])
    if "-h" in argv or "--help" in argv:
        _print_help()
        return
    legacy_mode = _token_itl_requested(argv) and not has_native_custom_spec_registry()
    if legacy_mode:
        argv = _rewrite_token_itl_to_ngram(argv)
        argv = _ensure_legacy_ngram_flags(argv)
        os.environ[LEGACY_PATCH_ENV] = "1"
        patch_legacy_ngram_worker()

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(argv)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def _token_itl_requested(argv: Sequence[str]) -> bool:
    for index, item in enumerate(argv):
        if item == "--speculative-algorithm":
            return index + 1 < len(argv) and argv[index + 1].upper() == TOKEN_ITL
        if item.startswith("--speculative-algorithm="):
            return item.split("=", 1)[1].upper() == TOKEN_ITL
    return False


def _rewrite_token_itl_to_ngram(argv: Sequence[str]) -> list[str]:
    rewritten = list(argv)
    for index, item in enumerate(rewritten):
        if item == "--speculative-algorithm" and index + 1 < len(rewritten):
            if rewritten[index + 1].upper() == TOKEN_ITL:
                rewritten[index + 1] = "NGRAM"
            return rewritten
        if item.startswith("--speculative-algorithm="):
            name = item.split("=", 1)[1]
            if name.upper() == TOKEN_ITL:
                rewritten[index] = "--speculative-algorithm=NGRAM"
            return rewritten
    return rewritten


def _ensure_legacy_ngram_flags(argv: Sequence[str]) -> list[str]:
    """Add NGRAM flags needed by SGLang 0.5.9 before its validation runs."""

    rewritten = list(argv)
    if not _has_option(rewritten, "--speculative-ngram-max-bfs-breadth"):
        rewritten += ["--speculative-ngram-max-bfs-breadth", "1"]
    if not _has_option(rewritten, "--disable-cuda-graph"):
        rewritten.append("--disable-cuda-graph")
    if not _has_option(rewritten, "--disable-overlap-schedule"):
        rewritten.append("--disable-overlap-schedule")
    return rewritten


def _has_option(argv: Sequence[str], option: str) -> bool:
    prefix = option + "="
    return any(item == option or item.startswith(prefix) for item in argv)


def _print_help() -> None:
    print(
        "usage: sglang-itl-launch [SGLang launch_server args]\n\n"
        "Launch SGLang with sglang-itl TOKEN_ITL compatibility. Pass the same\n"
        "arguments you would pass to `python -m sglang.launch_server`. On\n"
        "SGLang 0.5.9/0.5.10 this wrapper rewrites\n"
        "`--speculative-algorithm TOKEN_ITL` to the built-in NGRAM parser path\n"
        "and patches the worker factory before startup."
    )


if __name__ == "__main__":
    main()
