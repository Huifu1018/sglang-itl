"""Print or run production serving commands for MiniMax-M2.7-NVFP4."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokentiming.deployment import Engine, Mode, build_command, minimax_m27_nvfp4_profile, shell_join


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument(
        "--mode",
        choices=["baseline", "peagle", "eagle3", "standalone", "ngram", "token_itl"],
        default="peagle",
    )
    parser.add_argument("--target", default="nvidia/MiniMax-M2.7-NVFP4")
    parser.add_argument("--draft", default="phatv9/p-eagle-minimax-m2.7")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one raw engine CLI argument. Repeat for multiple arguments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = minimax_m27_nvfp4_profile(
        engine=args.engine,  # type: ignore[arg-type]
        mode=args.mode,  # type: ignore[arg-type]
        target_model=args.target,
        draft_model=args.draft,
        port=args.port,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
    )
    if args.extra_arg:
        profile = profile.__class__(**{**profile.__dict__, "extra_args": tuple(args.extra_arg)})

    command = build_command(profile)
    print(shell_join(command))

    if args.run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
