import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tokentiming.sglang.candidates import build_linear_candidate_rows
from tokentiming.sglang.config import TokenITLSGLangConfig
from tokentiming.sglang.validation import validate_server_args


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


if __name__ == "__main__":
    unittest.main()
