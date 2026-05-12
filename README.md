# Token-ITL

Token-ITL is a reference and deployment toolkit for heterogeneous-vocabulary
speculative decoding, focused on TokenTiming-style dynamic token alignment.

The repository includes:

- Dynamic Token Warping alignment for draft and target token streams.
- Draft top-1 probability projection onto target proxy tokens.
- A Hugging Face greedy verifier reference implementation.
- Deployment command builders for MiniMax-M2.7-NVFP4 serving profiles.
- OpenAI-compatible benchmark tooling.
- Documentation for universal draft models and production deployment choices.

## Install

Core tests use only the Python standard library.

For real model runs, install the optional runtime dependencies:

```bash
pip install -e ".[hf]"
```

## Test

```bash
python -m unittest discover -s tests -p 'test_tokentiming*.py'
python -m compileall tokentiming scripts
```

## Universal Draft Route

Use this route when:

- the target model and draft model have different tokenizers,
- the draft model is an ordinary off-the-shelf CausalLM,
- no target-specific EAGLE/P-EAGLE/MTP speculator is available.

See [docs/heterogeneous_vocab_universal_draft.md](docs/heterogeneous_vocab_universal_draft.md).

Pair benchmark:

```bash
python scripts/tokentiming_pair_bench.py \
  --target nvidia/MiniMax-M2.7-NVFP4 \
  --draft Qwen/Qwen2.5-1.5B-Instruct \
  --max-new-tokens 128 \
  --num-draft-tokens 4
```

## MiniMax-M2.7-NVFP4 Serving Profiles

Generate a vLLM command:

```bash
python scripts/minimax_m27_nvfp4_deploy.py \
  --engine vllm \
  --mode peagle \
  --tp 8 \
  --port 8000
```

See [docs/minimax_m27_nvfp4_production.md](docs/minimax_m27_nvfp4_production.md).

## Notes

The Hugging Face decoder in `tokentiming/hf_decoder.py` is a deterministic
greedy verifier reference. For high-throughput production, move the same
retokenization, alignment, and block-verification logic into the serving engine
worker so KV cache and scheduler state are reused.
