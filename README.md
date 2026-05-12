# Token-ITL

Token-ITL is a toolkit for heterogeneous-vocabulary speculative decoding. It is
designed for the case where the target model and draft model use different
tokenizers, and the draft model is an ordinary off-the-shelf causal language
model rather than a target-specific MTP, EAGLE, or P-EAGLE speculator.

The project currently implements a TokenTiming-style route:

```text
draft tokens
  -> decode to text
  -> retokenize with the target tokenizer
  -> Dynamic Token Warping alignment
  -> target block verification
```

It also includes deployment helpers and benchmark scripts for evaluating whether
a draft-target pair is actually worth using in production.

## Why This Exists

Speculative decoding is usually easiest when target and draft share a tokenizer.
That assumption does not hold for many practical model pairs. For example, a
large target such as MiniMax-M2.7-NVFP4 may need to be accelerated with a cheap
general draft model from another family.

Token-ITL focuses on that setting:

- target and draft vocabularies may differ,
- the draft model does not need target-specific training,
- candidate text is retokenized into target proxy tokens,
- Dynamic Token Warping aligns draft tokens and target proxy tokens,
- the target model verifies the proposed block.

## Features

- Dynamic Token Warping for heterogeneous token streams.
- Draft top-1 probability projection onto target proxy tokens.
- Hugging Face greedy verifier reference implementation.
- Pair benchmark for target/draft acceptance-rate analysis.
- vLLM/SGLang command generation for MiniMax-M2.7-NVFP4 serving profiles.
- OpenAI-compatible endpoint benchmark utility.
- Standard-library unit tests for the core algorithm and deployment builders.

## Repository Layout

```text
tokentiming/
  alignment.py       # DTW alignment and edit distance
  prob_mapping.py    # draft probability mapping and acceptance helpers
  hf_decoder.py      # Hugging Face greedy verifier reference
  deployment.py      # vLLM/SGLang serving command builders
  config.py          # runtime configuration
  result.py          # generation traces and stats
  tokenization.py    # tokenizer adapter

scripts/
  tokentiming_pair_bench.py        # target/draft pair benchmark
  minimax_m27_nvfp4_deploy.py      # MiniMax-M2.7-NVFP4 serving commands
  openai_compat_bench.py           # OpenAI-compatible endpoint benchmark

docs/
  tokentiming.md
  heterogeneous_vocab_universal_draft.md
  minimax_m27_nvfp4_production.md

configs/
  minimax_m27_nvfp4_tokentiming_universal.json
  minimax_m27_nvfp4_vllm_peagle.json
```

## Install

Core tests do not require PyTorch or Transformers.

```bash
git clone https://github.com/Huifu1018/Token-ITL.git
cd Token-ITL
python -m unittest discover -s tests -p 'test_tokentiming*.py'
```

For real model runs:

```bash
pip install -e ".[hf]"
```

This installs the optional Hugging Face runtime dependencies declared in
`pyproject.toml`.

## Quick Start: Heterogeneous Draft Pair

Use `tokentiming_pair_bench.py` to evaluate whether an ordinary draft model is
useful for a target model.

```bash
python scripts/tokentiming_pair_bench.py \
  --target nvidia/MiniMax-M2.7-NVFP4 \
  --draft Qwen/Qwen2.5-1.5B-Instruct \
  --max-new-tokens 128 \
  --num-draft-tokens 4 \
  --dtw-window 8
```

Key metrics:

- `acceptance_rate`: fraction of proxy target tokens accepted.
- `tokens_per_target_forward`: generated tokens divided by target forwards.
- `target_forwards`: number of verifier forwards.
- `draft_forwards`: number of draft forwards.
- `tokens_per_second`: wall-clock throughput for the reference implementation.

As a starting rule, keep a draft pair only if it reaches roughly:

```text
acceptance_rate >= 0.45
tokens_per_target_forward >= 1.25
```

These thresholds are workload-dependent, but they prevent spending engineering
time on pairs that are slower than target-only decoding.

## Python API

```python
from tokentiming import dynamic_token_warping, map_top1_draft_probabilities

alignment = dynamic_token_warping(
    draft_tokens=["a", "b"],
    target_tokens=["ab"],
    window=8,
)

mapped = map_top1_draft_probabilities(
    draft_token_ids=[10, 11],
    target_token_ids=[20],
    alignment=alignment,
    draft_token_probabilities=[0.6, 0.9],
)
```

Reference Hugging Face greedy verifier:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokentiming import TokenTimingConfig, TokenTimingGreedyDecoder

target_id = "nvidia/MiniMax-M2.7-NVFP4"
draft_id = "Qwen/Qwen2.5-1.5B-Instruct"

target_tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
draft_tokenizer = AutoTokenizer.from_pretrained(draft_id, trust_remote_code=True)
target_model = AutoModelForCausalLM.from_pretrained(
    target_id,
    device_map="auto",
    trust_remote_code=True,
)
draft_model = AutoModelForCausalLM.from_pretrained(
    draft_id,
    device_map="auto",
    trust_remote_code=True,
)

decoder = TokenTimingGreedyDecoder(
    target_model=target_model,
    draft_model=draft_model,
    target_tokenizer=target_tokenizer,
    draft_tokenizer=draft_tokenizer,
    config=TokenTimingConfig(max_new_tokens=128, num_draft_tokens=4),
)

result = decoder.generate("Explain speculative decoding in one paragraph.")
print(result.text)
print(result.stats.acceptance_rate)
```

The Hugging Face decoder is a correctness/reference implementation. For
high-throughput serving, move the same retokenization, alignment, and block
verification logic into the serving engine worker so KV cache and scheduling
state are reused.

## MiniMax-M2.7-NVFP4 Notes

If a trained speculator is available, a P-EAGLE/EAGLE route may be a better
production default. If the constraint is "ordinary draft model only, no
target-specific training", use the universal TokenTiming route documented in:

[docs/heterogeneous_vocab_universal_draft.md](docs/heterogeneous_vocab_universal_draft.md)

The repository also contains command builders for baseline/P-EAGLE/NGRAM
serving profiles:

```bash
python scripts/minimax_m27_nvfp4_deploy.py \
  --engine vllm \
  --mode peagle \
  --tp 8 \
  --port 8000
```

See:

- [docs/minimax_m27_nvfp4_production.md](docs/minimax_m27_nvfp4_production.md)
- [configs/minimax_m27_nvfp4_tokentiming_universal.json](configs/minimax_m27_nvfp4_tokentiming_universal.json)

## Benchmark an OpenAI-Compatible Server

After deploying a baseline server and a speculative server, compare them with:

```bash
python scripts/openai_compat_bench.py \
  --base-url http://127.0.0.1:8000 \
  --model nvidia/MiniMax-M2.7-NVFP4 \
  --requests 64 \
  --concurrency 4 \
  --max-tokens 512 \
  --temperature 0
```

The script reports latency, request QPS, output tokens/s, and failures.

## Tests

```bash
python -m unittest discover -s tests -p 'test_tokentiming*.py'
python -m compileall tokentiming scripts examples tests
```

Current tests cover:

- edit distance and Dynamic Token Warping,
- many-to-one and one-to-many probability mapping,
- config validation and generation stats,
- vLLM/SGLang deployment command builders.

## Current Scope

Implemented:

- deterministic greedy target verification,
- heterogeneous tokenizer retokenization,
- DTW alignment,
- top-1 draft probability mapping,
- deployment and benchmark helpers.

Not implemented as a production sampler:

- full stochastic lossless residual sampling inside vLLM/SGLang,
- optimized KV-cache engine integration,
- target-specific speculator training.

The next production step is to move the TokenTiming alignment and block
verification path into an inference engine worker instead of using the reference
Hugging Face loop.

## License

MIT.
