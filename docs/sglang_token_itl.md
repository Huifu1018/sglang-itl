# SGLang TOKEN_ITL Integration

This integration registers a SGLang speculative algorithm named `TOKEN_ITL`.
It targets ordinary draft models with a tokenizer different from the target
model tokenizer.

## Install

```bash
pip install -e ".[sglang]"
```

SGLang discovers the plugin through the `sglang.srt.plugins` entry point. To
load only this plugin when several SGLang plugins are installed:

```bash
export SGLANG_PLUGINS=token_itl
```

## Start A Server

```bash
export TOKEN_ITL_DRAFT_DEVICE=cuda:0
export TOKEN_ITL_DRAFT_DTYPE=bfloat16
export TOKEN_ITL_DTW_WINDOW=8

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

## Current Scope

The first SGLang path is correctness-first:

- Target verification, request mutation, and KV cache handling use SGLang's
  internal spec-v1 verification path.
- Draft candidates come from an ordinary HF `AutoModelForCausalLM` and are
  retokenized into target-vocabulary proxy tokens.
- Greedy decoding is supported. Sampling requires carrying proposal
  probabilities through SGLang's verifier and is not enabled yet.
- Overlap schedule, target CUDA graph, DP attention, pipeline parallelism, and
  multimodal requests are intentionally disabled or bypassed for this worker.
