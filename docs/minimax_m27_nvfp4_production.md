# MiniMax-M2.7-NVFP4 Speculative Decoding Production Plan

This plan targets `nvidia/MiniMax-M2.7-NVFP4` as the verifier model.

## Default Route

Use vLLM with a P-EAGLE/EAGLE3 speculator:

- Target: `nvidia/MiniMax-M2.7-NVFP4`
- Draft/speculator: `phatv9/p-eagle-minimax-m2.7`
- vLLM speculation method: `eagle3`
- Parallel drafting: enabled

This is the production-first route because the target is a large NVFP4 MoE model.
A normal small draft LLM plus TokenTiming DTW can work, but a target-specific
EAGLE/P-EAGLE speculator has a much better chance of giving stable acceptance.

## Generate the vLLM Command

```bash
sglang-itl-minimax-m27 \
  --engine vllm \
  --mode peagle \
  --target nvidia/MiniMax-M2.7-NVFP4 \
  --draft phatv9/p-eagle-minimax-m2.7 \
  --tp 8 \
  --max-model-len 32768 \
  --port 8000
```

The generated command has this shape:

```bash
vllm serve nvidia/MiniMax-M2.7-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --speculative-config '{"method":"eagle3","model":"phatv9/p-eagle-minimax-m2.7","num_speculative_tokens":5,"parallel_drafting":true}'
```

Add `--run` if you want the helper to execute the command directly.

## Alternative Draft Checkpoint

If `phatv9/p-eagle-minimax-m2.7` fails to load or gives poor acceptance, try:

```bash
sglang-itl-minimax-m27 \
  --engine vllm \
  --mode peagle \
  --draft nwzjk/p-eagle-minimax-m2.7 \
  --tp 8 \
  --port 8000
```

Both public checkpoints are experimental and should be load-tested before
production traffic.

## SGLang Route

SGLang is a strong serving option for NVFP4, especially on Blackwell hardware.
The helper can generate a SGLang EAGLE3 command:

```bash
sglang-itl-minimax-m27 \
  --engine sglang \
  --mode eagle3 \
  --target nvidia/MiniMax-M2.7-NVFP4 \
  --draft phatv9/p-eagle-minimax-m2.7 \
  --tp 8 \
  --max-model-len 32768 \
  --port 8000
```

Generated shape:

```bash
python3 -m sglang.launch_server \
  --model nvidia/MiniMax-M2.7-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tp 8 \
  --context-length 32768 \
  --quantization modelopt_fp4 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path phatv9/p-eagle-minimax-m2.7 \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6
```

Treat this as a compatibility route. P-EAGLE support is first-class in vLLM's
parallel drafting path; SGLang compatibility depends on whether the checkpoint
loads as an EAGLE3 speculator in your installed SGLang version.

## Benchmark

Run the same benchmark against baseline and speculative servers.

Baseline server:

```bash
sglang-itl-minimax-m27 \
  --engine vllm \
  --mode baseline \
  --tp 8 \
  --port 8000 \
  --run
```

Speculative server:

```bash
sglang-itl-minimax-m27 \
  --engine vllm \
  --mode peagle \
  --tp 8 \
  --port 8001 \
  --run
```

Benchmark:

```bash
sglang-itl-bench \
  --base-url http://127.0.0.1:8000 \
  --model nvidia/MiniMax-M2.7-NVFP4 \
  --requests 64 \
  --concurrency 4 \
  --max-tokens 512 \
  --temperature 0

sglang-itl-bench \
  --base-url http://127.0.0.1:8001 \
  --model nvidia/MiniMax-M2.7-NVFP4 \
  --requests 64 \
  --concurrency 4 \
  --max-tokens 512 \
  --temperature 0
```

Compare:

- `output_tokens_per_s`
- `latency_p50_s`
- `latency_p95_s`
- server logs for speculative acceptance length / acceptance rate
- GPU utilization and memory headroom

## Fallback Policy

Use this production fallback order:

1. vLLM + P-EAGLE: best target route.
2. vLLM + NGRAM: no draft checkpoint dependency, modest gain.
3. SGLang + EAGLE3: use if your SGLang build loads the speculator better.
4. vLLM baseline: always keep as the safe control plane fallback.

## Where TokenTiming Fits

TokenTiming is the right algorithm when the draft is a normal CausalLM with a
different tokenizer. For MiniMax-M2.7-NVFP4, the first production choice should
be a dedicated P-EAGLE speculator. If no speculator is usable, TokenTiming's DTW
alignment code can be moved into an engine worker to support universal draft
models, but that requires engine-level access to candidate logits and target
verification logits.

The included `tokentiming` module provides the reusable pieces for that path:

- `dynamic_token_warping`
- `map_top1_draft_probabilities`
- `TokenTimingGreedyDecoder`

Do not deploy the HF greedy decoder as the high-throughput service. It is a
reference implementation; production belongs inside vLLM/SGLang.
