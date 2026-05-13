import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

from tokentiming.sglang.candidates import build_linear_candidate_rows
from tokentiming.sglang.config import TokenITLSGLangConfig
from tokentiming.sglang.proposer import _clone_nested_tensors
from tokentiming.sglang.validation import validate_server_args
from tokentiming.cli.sglang_token_itl_launch import (
    _ensure_legacy_ngram_flags,
    _rewrite_token_itl_to_ngram,
    _token_itl_requested,
)


class TokenITLSGLangValidationTest(unittest.TestCase):
    def test_candidate_rows_shrink_to_shortest_real_chain(self):
        candidates = build_linear_candidate_rows(
            roots=[10, 20],
            proxy_rows=[[11, 12, 13], [21]],
            max_draft_token_num=4,
        )

        self.assertEqual(candidates.draft_token_num, 2)
        self.assertEqual(candidates.rows, ((10, 11), (20, 21)))
        self.assertEqual(candidates.proposed_proxy_tokens, 4)

    def test_candidate_rows_allow_target_only_width(self):
        candidates = build_linear_candidate_rows(
            roots=[10, 20],
            proxy_rows=[[11, 12], []],
            max_draft_token_num=4,
        )

        self.assertEqual(candidates.draft_token_num, 1)
        self.assertEqual(candidates.rows, ((10,), (20,)))

    def test_config_can_disable_periodic_metrics(self):
        with patch.dict("os.environ", {"TOKEN_ITL_METRICS_LOG_INTERVAL": "0"}):
            config = TokenITLSGLangConfig.from_env()

        self.assertIsNone(config.metrics_log_interval)

    def test_config_defaults_to_cloned_draft_cache(self):
        config = TokenITLSGLangConfig.from_env()

        self.assertTrue(config.clone_draft_cache)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_nested_tensor_clone_does_not_alias(self):
        source = ((torch.tensor([1, 2]), torch.tensor([3])),)
        cloned = _clone_nested_tensors(source)
        source[0][0][0] = 99

        self.assertEqual(int(cloned[0][0][0]), 1)

    def test_validation_sets_spec_v1_defaults(self):
        args = SimpleNamespace(
            speculative_draft_model_path="draft",
            enable_dp_attention=False,
            pp_size=1,
            device="cuda",
            max_running_requests=None,
            disable_overlap_schedule=False,
            enable_mixed_chunk=True,
            disable_cuda_graph=False,
            speculative_num_steps=None,
            speculative_eagle_topk=None,
            speculative_num_draft_tokens=None,
        )

        validate_server_args(args)

        self.assertTrue(args.disable_overlap_schedule)
        self.assertFalse(args.enable_mixed_chunk)
        self.assertTrue(args.disable_cuda_graph)
        self.assertEqual(args.speculative_num_steps, 4)
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.speculative_num_draft_tokens, 5)
        self.assertEqual(args.max_running_requests, 48)

    def test_validation_requires_draft_model(self):
        args = SimpleNamespace(
            speculative_draft_model_path=None,
            enable_dp_attention=False,
            pp_size=1,
            device="cuda",
        )

        with self.assertRaises(ValueError):
            validate_server_args(args)

    def test_legacy_launcher_rewrites_token_itl_to_ngram(self):
        args = ["--model-path", "target", "--speculative-algorithm", "TOKEN_ITL"]

        self.assertTrue(_token_itl_requested(args))
        rewritten = _rewrite_token_itl_to_ngram(args)

        self.assertEqual(rewritten[-1], "NGRAM")

    def test_legacy_launcher_adds_required_ngram_flags(self):
        args = ["--speculative-algorithm", "NGRAM"]

        rewritten = _ensure_legacy_ngram_flags(args)

        self.assertIn("--speculative-ngram-max-bfs-breadth", rewritten)
        self.assertIn("1", rewritten)
        self.assertIn("--disable-cuda-graph", rewritten)
        self.assertIn("--disable-overlap-schedule", rewritten)


if __name__ == "__main__":
    unittest.main()
