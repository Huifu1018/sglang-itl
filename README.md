# sglang-itl

sglang-itl is a toolkit for heterogeneous-vocabulary speculative decoding. It is
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

sglang-itl focuses on that setting:

- target and draft vocabularies may differ,
- the draft model does not need target-specific training,
- candidate text is retokenized into target proxy tokens,
- Dynamic Token Warping aligns draft tokens and target proxy tokens,
- the target model verifies the proposed block.

## Features

- Dynamic Token Warping for heterogeneous token streams.
- Draft top-1 probability projection onto target proxy tokens.
- Hugging Face greedy verifier reference implementation.
- SGLang `TOKEN_ITL` plugin for engine-level target verification.
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
  sglang/            # SGLang TOKEN_ITL plugin and worker

scripts/
  tokentiming_pair_bench.py        # target/draft pair benchmark
  minimax_m27_nvfp4_deploy.py      # MiniMax-M2.7-NVFP4 serving commands
  openai_compat_bench.py           # OpenAI-compatible endpoint benchmark

docs/
  tokentiming.md
  sglang_token_itl.md
  heterogeneous_vocab_universal_draft.md
  minimax_m27_nvfp4_production.md

configs/
  minimax_m27_nvfp4_tokentiming_universal.json
  minimax_m27_nvfp4_vllm_peagle.json
```

## Install

For the lightweight core package:

```bash
uv pip install sglang-itl
# or
pip install sglang-itl
```

For SGLang engine integration, install the package with the `sglang` extra:

```bash
uv pip install "sglang-itl[sglang]"
# or
pip install "sglang-itl[sglang]"
```

For HF-only pair evaluation without SGLang:

```bash
uv pip install "sglang-itl[hf]"
# or
pip install "sglang-itl[hf]"
```

Until the first PyPI release is published, install directly from GitHub:

```bash
uv pip install "sglang-itl[sglang] @ git+https://github.com/Huifu1018/sglang-itl.git"
# or
pip install "sglang-itl[sglang] @ git+https://github.com/Huifu1018/sglang-itl.git"
```

For local development:

```bash
git clone https://github.com/Huifu1018/sglang-itl.git
cd sglang-itl
python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[sglang]"
python -m unittest discover -s tests -p 'test_tokentiming*.py'
```

Core tests do not require PyTorch, Transformers, or SGLang.

## Usage Guide

### 1. Decide Which Path You Need

Use this table to pick the correct entry point.

| Goal | Command or Module |
| --- | --- |
| Check that the repo works | `python -m unittest discover -s tests -p 'test_tokentiming*.py'` |
| Evaluate a target/draft pair | `sglang-itl-pair-bench` |
| Run one prompt through the HF reference verifier | `examples/tokentiming_hf_demo.py` |
| Serve with SGLang engine-level verification | `--speculative-algorithm TOKEN_ITL` |
| Check TOKEN_ITL deployment readiness | `sglang-itl-preflight` |
| Generate MiniMax-M2.7-NVFP4 serving commands | `sglang-itl-minimax-m27` |
| Benchmark an already running OpenAI-compatible server | `sglang-itl-bench` |
| Import the algorithm in Python | `tokentiming.dynamic_token_warping` and `tokentiming.map_top1_draft_probabilities` |

If your draft model is a normal off-the-shelf LLM with a different tokenizer,
start with `sglang-itl-pair-bench`. If you already have a running vLLM/SGLang
server, use `sglang-itl-bench` to compare endpoint throughput.

### 2. Prepare Prompts

Create a plain text prompt file. Each non-empty line is one benchmark request.

```bash
cat > prompts.txt <<'EOF'
用中文解释 speculative decoding 的核心思想，限制在 200 字以内。
Write a concise Python function for binary search.
Compare vLLM and SGLang for serving large MoE models.
EOF
```

Use prompts that match your real traffic. Acceptance rate can change a lot
between chat, code, Chinese, English, and tool-use workloads.

### 3. Benchmark a Heterogeneous Target/Draft Pair

Run the pair benchmark. This uses the Hugging Face reference verifier, so it is
for correctness and pair selection, not final serving throughput.

```bash
sglang-itl-pair-bench \
  --target nvidia/MiniMax-M2.7-NVFP4 \
  --draft Qwen/Qwen2.5-1.5B-Instruct \
  --prompts-file prompts.txt \
  --max-new-tokens 128 \
  --num-draft-tokens 4 \
  --dtw-window 8 \
  --device-map auto
```

Important arguments:

- `--target`: verifier model. This is the model whose output must be preserved.
- `--draft`: ordinary draft model. It can use a different tokenizer.
- `--prompts-file`: benchmark prompts, one per line.
- `--max-new-tokens`: generated tokens per prompt.
- `--num-draft-tokens`: draft block size. Start with `4`; try `2`, `4`, `6`, `8`.
- `--dtw-window`: DTW alignment window. Start with `8`; increase if token counts differ heavily.
- `--device-map`: passed to Hugging Face `from_pretrained`.

Example output shape:

```json
{
  "prompts": 3,
  "generated_tokens": 384,
  "draft_forwards": 512,
  "target_forwards": 210,
  "proposed_proxy_tokens": 620,
  "accepted_proxy_tokens": 330,
  "elapsed_seconds": 42.3,
  "acceptance_rate": 0.5323,
  "tokens_per_target_forward": 1.8286,
  "tokens_per_second": 9.078
}
```

How to read it:

- Keep the pair if `acceptance_rate` is high enough and `tokens_per_target_forward` is above target-only decoding.
- Reject the pair if `tokens_per_target_forward <= 1.0`; it is not reducing verifier work.
- Compare several draft models on the same prompt file before choosing one.
- A larger `num_draft_tokens` is not automatically better; it can lower acceptance and waste draft work.

Practical first thresholds:

```text
acceptance_rate >= 0.45
tokens_per_target_forward >= 1.25
```

### 4. Run One Prompt With the HF Reference Verifier

Use this when you want to inspect generated text and trace behavior rather than
run a benchmark.

```bash
python examples/tokentiming_hf_demo.py \
  --target nvidia/MiniMax-M2.7-NVFP4 \
  --draft Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "用中文解释 TokenTiming 如何处理异构 tokenizer。" \
  --max-new-tokens 128 \
  --num-draft-tokens 4
```

The script prints the generated text and summary stats:

- `generated_tokens`
- `target_forwards`
- `draft_forwards`
- `acceptance_rate`
- `tokens_per_target_forward`

The HF verifier is intentionally simple. It proves the algorithmic path, but
production serving should move the same logic into vLLM/SGLang internals.

### 5. Generate Serving Commands

For MiniMax-M2.7-NVFP4 serving profiles:

```bash
sglang-itl-minimax-m27 \
  --engine vllm \
  --mode baseline \
  --tp 8 \
  --port 8000
```

For a P-EAGLE/EAGLE profile:

```bash
sglang-itl-minimax-m27 \
  --engine vllm \
  --mode peagle \
  --draft phatv9/p-eagle-minimax-m2.7 \
  --tp 8 \
  --port 8001
```

For SGLang:

```bash
sglang-itl-minimax-m27 \
  --engine sglang \
  --mode eagle3 \
  --draft phatv9/p-eagle-minimax-m2.7 \
  --tp 8 \
  --port 8001
```

Use `--run` only when you want the helper to execute the printed command.

### 6. Run SGLang TOKEN_ITL

`TOKEN_ITL` is an out-of-tree SGLang plugin. It uses an ordinary HF draft model
to propose text, retokenizes that text with the target tokenizer, then verifies
the target proxy tokens through SGLang's internal spec-v1 target verifier.

Install this package in the same environment as SGLang:

```bash
uv pip install "sglang-itl[sglang]"
export SGLANG_PLUGINS=token_itl
```

Run preflight on the deployment host before starting the server:

```bash
sglang-itl-preflight \
  --target nvidia/MiniMax-M2.7-NVFP4 \
  --draft Qwen/Qwen2.5-1.5B-Instruct
```

Start SGLang with the custom algorithm:

```bash
export TOKEN_ITL_DRAFT_DEVICE=cuda:0
export TOKEN_ITL_DRAFT_DTYPE=bfloat16
export TOKEN_ITL_DTW_WINDOW=8
export TOKEN_ITL_ENABLE_DRAFT_CACHE=true
export TOKEN_ITL_CLONE_DRAFT_CACHE=true
export TOKEN_ITL_MAX_CACHED_REQUESTS=256

python -m sglang.launch_server \
  --model-path nvidia/MiniMax-M2.7-NVFP4 \
  --trust-remote-code \
  --tp 8 \
  --quantization modelopt_fp4 \
  --speculative-algorithm TOKEN_ITL \
  --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
  --speculative-num-steps 4 \
  --speculative-num-draft-tokens 5 \
  --disable-overlap-schedule \
  --disable-cuda-graph
```

Use greedy requests:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/MiniMax-M2.7-NVFP4",
    "messages": [{"role": "user", "content": "解释 TokenTiming 的异构词表验证流程"}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

The deployment helper can print the same shape of command:

```bash
sglang-itl-minimax-m27 \
  --engine sglang \
  --mode token_itl \
  --draft Qwen/Qwen2.5-1.5B-Instruct \
  --tp 8 \
  --port 30000
```

Current SGLang scope:

- greedy decoding only (`temperature=0`),
- text-only requests,
- no overlap scheduler, target CUDA graph, DP attention, or pipeline parallelism,
- target verification and KV mutation are inside SGLang,
- draft proposal uses an ordinary HF draft model with per-request
  `past_key_values` cache and conservative prefix validation.

Full details: [docs/sglang_token_itl.md](docs/sglang_token_itl.md)

### 7. Benchmark a Running Server

After starting a vLLM/SGLang OpenAI-compatible endpoint, benchmark it with:

```bash
sglang-itl-bench \
  --base-url http://127.0.0.1:8000 \
  --model nvidia/MiniMax-M2.7-NVFP4 \
  --prompts-file prompts.txt \
  --requests 64 \
  --concurrency 4 \
  --max-tokens 512 \
  --temperature 0
```

Compare baseline and speculative servers using:

- `output_tokens_per_s`
- `latency_p50_s`
- `latency_p95_s`
- `failed`

This endpoint benchmark cannot implement TokenTiming by itself. TokenTiming
needs engine-level access to draft logits, retokenized proxy tokens, and target
verification logits. Use this script to measure a server after integration.

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
sglang-itl-minimax-m27 \
  --engine vllm \
  --mode peagle \
  --tp 8 \
  --port 8000
```

See:

- [docs/minimax_m27_nvfp4_production.md](docs/minimax_m27_nvfp4_production.md)
- [configs/minimax_m27_nvfp4_tokentiming_universal.json](configs/minimax_m27_nvfp4_tokentiming_universal.json)

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
- SGLang `TOKEN_ITL` plugin using SGLang's internal target verifier and
  per-request draft KV cache,
- deployment and benchmark helpers.

Not implemented as a production sampler:

- full stochastic lossless residual sampling inside vLLM/SGLang,
- native SGLang `ModelRunner` draft execution for heterogeneous-tokenizer draft
  models,
- target-specific speculator training.

The next production step is to replace the HF draft runner with a native SGLang
draft worker while preserving the heterogeneous-tokenizer retokenization and
verification contract.

## License

MIT.
