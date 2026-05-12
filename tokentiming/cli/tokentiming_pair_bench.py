"""Benchmark a heterogeneous target/draft pair with TokenTiming greedy verify."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokentiming import TokenTimingConfig, TokenTimingGreedyDecoder


DEFAULT_PROMPTS = [
    "用中文解释 speculative decoding 的核心思想，限制在 200 字以内。",
    "Write a concise Python implementation of binary search.",
    "请比较 vLLM 和 SGLang 在大 MoE 模型部署上的差异。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument("--dtw-window", type=int, default=8)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError("prompts-file contains no prompts.")
    return prompts


def load_model_and_tokenizer(model_id: str, args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.torch_dtype != "auto":
        import torch

        model_kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return model, tokenizer


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    target_model, target_tokenizer = load_model_and_tokenizer(args.target, args)
    draft_model, draft_tokenizer = load_model_and_tokenizer(args.draft, args)

    decoder = TokenTimingGreedyDecoder(
        target_model=target_model,
        draft_model=draft_model,
        target_tokenizer=target_tokenizer,
        draft_tokenizer=draft_tokenizer,
        config=TokenTimingConfig(
            max_new_tokens=args.max_new_tokens,
            num_draft_tokens=args.num_draft_tokens,
            dtw_window=args.dtw_window,
            eos_token_id=target_tokenizer.eos_token_id,
            device=args.input_device,
        ),
    )

    totals = {
        "prompts": len(prompts),
        "generated_tokens": 0,
        "draft_forwards": 0,
        "target_forwards": 0,
        "proposed_proxy_tokens": 0,
        "accepted_proxy_tokens": 0,
        "elapsed_seconds": 0.0,
    }
    for prompt in prompts:
        result = decoder.generate(prompt)
        totals["generated_tokens"] += result.stats.generated_tokens
        totals["draft_forwards"] += result.stats.draft_forwards
        totals["target_forwards"] += result.stats.target_forwards
        totals["proposed_proxy_tokens"] += result.stats.proposed_proxy_tokens
        totals["accepted_proxy_tokens"] += result.stats.accepted_proxy_tokens
        totals["elapsed_seconds"] += result.stats.elapsed_seconds

    acceptance_rate = (
        totals["accepted_proxy_tokens"] / totals["proposed_proxy_tokens"]
        if totals["proposed_proxy_tokens"]
        else 0.0
    )
    tokens_per_target_forward = (
        totals["generated_tokens"] / totals["target_forwards"]
        if totals["target_forwards"]
        else 0.0
    )
    tokens_per_second = (
        totals["generated_tokens"] / totals["elapsed_seconds"]
        if totals["elapsed_seconds"]
        else 0.0
    )
    summary = {
        **totals,
        "acceptance_rate": round(acceptance_rate, 4),
        "tokens_per_target_forward": round(tokens_per_target_forward, 4),
        "tokens_per_second": round(tokens_per_second, 4),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
