# SGLang TOKEN_ITL Integration

This integration registers a SGLang speculative algorithm named `TOKEN_ITL`.
It targets ordinary draft models with a tokenizer different from the target
model tokenizer.

SGLang 0.5.9 and 0.5.10 do not expose the native out-of-tree speculative plugin
registry. For those versions, use `sglang-itl-launch`; it rewrites
`TOKEN_ITL` to the built-in NGRAM spec-v1 parser path and patches the worker
factory before SGLang starts.

## Install

```bash
uv pip install "sglang-itl[sglang]"
# or
pip install "sglang-itl[sglang]"
```

For a deployment pinned to SGLang 0.5.9:

```bash
uv pip install "sglang==0.5.9" "sglang-itl[sglang]"
```

Before the first PyPI release, install from GitHub:

```bash
uv pip install "sglang-itl[sglang] @ git+https://github.com/Huifu1018/sglang-itl.git"
```

On SGLang versions with native custom speculative plugins, SGLang discovers the
plugin through the `sglang.srt.plugins` entry point. To load only this plugin
when several SGLang plugins are installed:

```bash
export SGLANG_PLUGINS=token_itl
```

## Preflight

Run this on the deployment host. It verifies the installed package, SGLang
plugin entry point, `TOKEN_ITL` registry, CUDA availability, and optional
target/draft model configs without loading model weights.

```bash
sglang-itl-preflight \
  --target nvidia/MiniMax-M2.7-NVFP4 \
  --draft Qwen/Qwen2.5-1.5B-Instruct
```

On a non-GPU development machine, use `--allow-no-cuda` only to validate the
Python/package wiring.

## Start A Server

```bash
export TOKEN_ITL_DRAFT_DEVICE=cuda:0
export TOKEN_ITL_DRAFT_DTYPE=bfloat16
export TOKEN_ITL_DTW_WINDOW=8
export TOKEN_ITL_ENABLE_DRAFT_CACHE=true
export TOKEN_ITL_CLONE_DRAFT_CACHE=true
export TOKEN_ITL_MAX_CACHED_REQUESTS=256

sglang-itl-launch \
  --model-path nvidia/MiniMax-M2.7-NVFP4 \
  --trust-remote-code \
  --tp 8 \
  --quantization modelopt_fp4 \
  --speculative-algorithm TOKEN_ITL \
  --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
  --speculative-num-steps 4 \
  --speculative-num-draft-tokens 5 \
  --speculative-ngram-max-bfs-breadth 1 \
  --disable-overlap-schedule \
  --disable-cuda-graph
```

Requests should use greedy decoding:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/MiniMax-M2.7-NVFP4",
    "messages": [{"role": "user", "content": "解释一下异构词表 speculative decoding"}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

## Runtime Knobs

`TOKEN_ITL_DRAFT_DEVICE`: device for the HF draft model, defaulting to the
SGLang worker GPU.

`TOKEN_ITL_DRAFT_DEVICE_MAP`: optional Transformers `device_map`; when set,
the draft model is not moved with `.to(device)`.

`TOKEN_ITL_DRAFT_DTYPE`: `auto`, `float16`, `bfloat16`, or `float32`.

`TOKEN_ITL_DTW_WINDOW`: DTW alignment window for TokenTiming traces.

`TOKEN_ITL_MAX_DRAFT_TOKENS`: upper bound for draft-token generation while
trying to collect enough target proxy tokens.

`TOKEN_ITL_MAX_CONTEXT_TOKENS`: optional draft-side context window. Leave unset
for full draft context. Set it when the ordinary draft model has a shorter
context window than the target.

`TOKEN_ITL_ENABLE_DRAFT_CACHE`: keep per-request HF `past_key_values` caches.
Default: `true`.

`TOKEN_ITL_CLONE_DRAFT_CACHE`: clone/fork cached draft KV before speculative
proposal generation. Default: `true`, which prevents unaccepted draft tokens
from mutating the confirmed per-request context cache on HF cache
implementations that update in place.

`TOKEN_ITL_MAX_CACHED_REQUESTS`: LRU cap for draft request caches. Default:
`256`.

`TOKEN_ITL_METRICS_LOG_INTERVAL`: seconds between worker metric log lines.
Default: `60`. Set it to `0` to disable periodic metric logging.

## Execution Path

For every decode batch:

1. Reconstruct each request's current target text from SGLang request state.
2. Encode that text with the draft tokenizer.
3. Reuse a per-request draft KV cache when the newly encoded draft context has
   the cached draft token ids as an exact prefix.
4. Fork the cached draft KV and generate a short greedy draft block with the
   ordinary HF draft model.
5. Decode the draft block to text and retokenize it with the target tokenizer.
6. Build one linear target-token candidate chain per request.
7. Verify the candidate chains through SGLang's internal target verifier.
8. Evict draft cache state for finished or retracted requests.

The worker does not pad missing candidates into real target tokens. If any
request in a batch cannot produce enough valid proxy tokens, the whole batch
verify width shrinks to the shortest real candidate chain. If the shortest
chain has no draft candidate, SGLang performs a target-only decode step for that
batch.

## Current Scope

This SGLang path is engine-integrated but intentionally conservative:

- Target verification, request mutation, and KV cache handling use SGLang's
  internal spec-v1 verification path.
- Draft candidates come from an ordinary HF `AutoModelForCausalLM` and are
  retokenized into target-vocabulary proxy tokens.
- The HF draft side keeps per-request `past_key_values` caches and falls back to
  rebuild when tokenizer boundary effects invalidate prefix reuse.
- Greedy decoding is supported. Sampling requires carrying proposal
  probabilities through SGLang's verifier and is not enabled yet.
- Overlap schedule, target CUDA graph, DP attention, pipeline parallelism, and
  multimodal requests are intentionally disabled or bypassed for this worker.
