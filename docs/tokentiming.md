# TokenTiming Implementation Notes

This directory contains an independent TokenTiming-style implementation for
heterogeneous-tokenizer speculative decoding.

## What Is Implemented

- Dynamic Token Warping (DTW) alignment between generated draft tokens and
  retokenized target proxy tokens.
- Draft top-1 probability projection onto target proxy tokens.
- A Hugging Face greedy verifier that checks a full proxy block with one target
  model forward.
- Structured traces and aggregate stats for acceptance-rate analysis.

The greedy verifier is deterministic and target-equivalent: accepted proxy
tokens are only appended when they match the target model's greedy next token.
On mismatch, the target model's greedy token is appended instead.

## What Is Deliberately Not Implemented

Lossless stochastic speculative sampling is not exposed in this module. A
production-quality sampler needs a full proposal distribution over the target
vocabulary to form the residual distribution after rejection. TokenTiming's
top-1 mapped probability is enough for diagnostics and paper-style acceptance
analysis, but it is not enough by itself for a conservative industrial lossless
sampler.

## Install Runtime Dependencies

The core alignment and mapping tests use only the Python standard library. To
run real models, install at least:

```bash
pip install torch transformers accelerate sentencepiece safetensors
```

## Run the Example

```bash
python examples/tokentiming_hf_demo.py \
  --target Qwen/Qwen2.5-1.5B-Instruct \
  --draft Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain speculative decoding in one paragraph." \
  --max-new-tokens 64 \
  --num-draft-tokens 8
```

The script prints the generated text plus:

- generated target tokens
- target forward count
- draft forward count
- proxy-token acceptance rate
- generated tokens per target forward

## Integration With the Previous Paper's Code

The previous heterogeneous-vocabulary speculative decoding implementation is a
good base for:

- candidate generation loops
- target verification scheduling
- tokenizer retokenization plumbing
- KV cache management
- benchmark harnesses

For TokenTiming, replace the SLEM/TLI alignment component with:

```python
from tokentiming import dynamic_token_warping, map_top1_draft_probabilities
```

The rest of the serving engine should keep responsibility for batching, cache
reuse, request scheduling, and target-model fallback.

## Production Checklist

- Keep `num_draft_tokens` small at first, usually 4 to 8.
- Track `acceptance_rate` and `tokens_per_target_forward`; bad draft pairs can
  be slower than plain target greedy decoding.
- Use a draft model that is much cheaper than the target model and behaviorally
  close to it.
- For vLLM/SGLang integration, move DTW and probability mapping into the
  existing speculative decoding worker instead of retokenizing the full current
  text on every block.
- Treat heterogeneous tokenizer special tokens carefully. If the two models use
  different EOS strings, prefer engine-level EOS handling over text round-trip
  handling.
