"""Example wiring for the TokenTiming greedy prototype.

Usage:
    python examples/tokentiming_hf_demo.py \
      --target Qwen/Qwen2.5-1.5B-Instruct \
      --draft Qwen/Qwen2.5-0.5B-Instruct \
      --prompt "Explain speculative decoding in one paragraph."
"""

from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from tokentiming import TokenTimingConfig, TokenTimingGreedyDecoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-draft-tokens", type=int, default=8)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--input-device", default=None)
    args = parser.parse_args()

    target_tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    eos_token_id = target_tokenizer.eos_token_id
    decoder = TokenTimingGreedyDecoder(
        target_model=target_model,
        draft_model=draft_model,
        target_tokenizer=target_tokenizer,
        draft_tokenizer=draft_tokenizer,
        config=TokenTimingConfig(
            max_new_tokens=args.max_new_tokens,
            num_draft_tokens=args.num_draft_tokens,
            eos_token_id=eos_token_id,
            device=args.input_device,
        ),
    )
    result = decoder.generate(args.prompt)
    print(result.text)
    print()
    print("TokenTiming stats:")
    print(f"  generated_tokens={result.stats.generated_tokens}")
    print(f"  target_forwards={result.stats.target_forwards}")
    print(f"  draft_forwards={result.stats.draft_forwards}")
    print(f"  acceptance_rate={result.stats.acceptance_rate:.3f}")
    print(f"  tokens_per_target_forward={result.stats.tokens_per_target_forward:.3f}")


if __name__ == "__main__":
    main()
