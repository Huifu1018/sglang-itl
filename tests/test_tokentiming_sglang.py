import unittest
from types import SimpleNamespace

from tokentiming.sglang.validation import validate_server_args


class TokenITLSGLangValidationTest(unittest.TestCase):
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
