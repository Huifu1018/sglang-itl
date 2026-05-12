# Heterogeneous-Vocabulary Speculative Decoding With an Untrained Draft

This is the route for:

- a fixed target model,
- an ordinary off-the-shelf draft model,
- different tokenizers/vocabularies,
- no target-specific draft training.

For `MiniMax-M2.7-NVFP4`, this is the relevant route if we deliberately exclude
MTP, EAGLE, P-EAGLE, or any trained speculator.

## Method Choice

Use TokenTiming as the primary method.

TokenTiming explicitly targets universal speculative decoding model pairs: it
decodes draft tokens to text, re-encodes that text with the target tokenizer,
then uses DTW-style alignment to transfer draft probabilities for speculative
verification.

Use the previous heterogeneous-vocabulary paper as the conservative baseline:

- SLEM: simplest exact string-level matching baseline.
- TLI: useful when the two tokenizers share enough token-level overlap.
- SLRS: theoretically important, but more expensive to make practical at scale.

## Serving Flow

```text
prompt text
  -> draft tokenizer
  -> draft model proposes k tokens
  -> decode draft continuation to text
  -> target tokenizer retokenizes continuation into proxy target tokens
  -> TokenTiming DTW aligns draft tokens and proxy target tokens
  -> target model verifies the proxy block in one forward
  -> accept matching prefix, otherwise append target fallback token
```

The current implementation provides this flow for greedy decoding in:

```python
from tokentiming import TokenTimingConfig, TokenTimingGreedyDecoder
```

This decoder is target-greedy-equivalent. For stochastic lossless sampling, the
engine integration must implement the paper's full acceptance and residual
sampling logic.

## Draft Model Selection

Do not choose the draft by parameter count alone. The draft must be cheap and
behaviorally close to the target's expected output distribution.

Start with 3 to 5 candidates:

- one small general chat model,
- one code-strong model if serving code workloads,
- one Chinese/English bilingual model if serving mixed zh/en traffic,
- one very small model as a latency lower bound,
- one larger draft as an acceptance upper bound.

Keep a candidate only if it improves both:

- proxy-token acceptance rate,
- generated tokens per target forward.

Initial thresholds:

- acceptance rate >= 0.45
- tokens per target forward >= 1.25

If a draft misses both thresholds, it is slower than it looks.

## Production Integration

The HF decoder in this repo is a correctness/reference implementation. For
production, move the same steps into the serving engine:

- draft worker generates candidate tokens and draft logits,
- tokenizer bridge retokenizes only the new continuation, not the whole prompt,
- target worker verifies the full proxy block with KV cache,
- scheduler records accepted length and falls back to ordinary target decoding
  when acceptance drops.

An OpenAI-compatible HTTP API is not enough for the verifier path, because the
server must inspect logits for proposed proxy token positions.

## Pair Benchmark

Use the pair benchmark before any production integration:

```bash
sglang-itl-pair-bench \
  --target nvidia/MiniMax-M2.7-NVFP4 \
  --draft Qwen/Qwen2.5-1.5B-Instruct \
  --prompts-file prompts.txt \
  --max-new-tokens 128 \
  --num-draft-tokens 4
```

Compare candidate drafts using:

- `acceptance_rate`
- `tokens_per_target_forward`
- `target_forwards`
- `draft_forwards`
- wall-clock tokens/s

Only after this benchmark should the pair be moved into vLLM/SGLang internals.
