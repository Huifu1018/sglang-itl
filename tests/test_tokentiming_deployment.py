import json
import unittest

from tokentiming.deployment import (
    ServingProfile,
    build_sglang_command,
    build_vllm_command,
    minimax_m27_nvfp4_profile,
)


class TokenTimingDeploymentTest(unittest.TestCase):
    def test_vllm_peagle_command_contains_parallel_drafting(self):
        profile = minimax_m27_nvfp4_profile(
            engine="vllm",
            mode="peagle",
            tensor_parallel_size=8,
        )
        command = build_vllm_command(profile)

        self.assertIn("vllm", command)
        self.assertIn("nvidia/MiniMax-M2.7-NVFP4", command)
        spec = json.loads(command[command.index("--speculative-config") + 1])
        self.assertEqual(spec["method"], "eagle3")
        self.assertEqual(spec["model"], "phatv9/p-eagle-minimax-m2.7")
        self.assertTrue(spec["parallel_drafting"])

    def test_vllm_ngram_command_has_no_draft_model(self):
        profile = minimax_m27_nvfp4_profile(engine="vllm", mode="ngram")
        command = build_vllm_command(profile)
        spec = json.loads(command[command.index("--speculative-config") + 1])

        self.assertEqual(spec["method"], "ngram")
        self.assertNotIn("model", spec)

    def test_sglang_eagle3_command_contains_modelopt_fp4(self):
        profile = minimax_m27_nvfp4_profile(
            engine="sglang",
            mode="eagle3",
            tensor_parallel_size=8,
        )
        command = build_sglang_command(profile)

        self.assertIn("--quantization", command)
        self.assertIn("modelopt_fp4", command)
        self.assertIn("--speculative-algorithm", command)
        self.assertIn("EAGLE3", command)

    def test_sglang_token_itl_command_registers_custom_algorithm(self):
        profile = ServingProfile(
            engine="sglang",
            mode="token_itl",
            target_model="target",
            draft_model="draft",
            speculative_num_steps=4,
            speculative_num_draft_tokens=5,
        )
        command = build_sglang_command(profile)

        self.assertIn("--speculative-algorithm", command)
        self.assertIn("TOKEN_ITL", command)
        self.assertIn("--speculative-draft-model-path", command)
        self.assertIn("draft", command)
        self.assertIn("--disable-overlap-schedule", command)
        self.assertIn("--disable-cuda-graph", command)

    def test_profile_validation_requires_draft(self):
        profile = ServingProfile(
            engine="vllm",
            mode="peagle",
            target_model="target",
            draft_model=None,
        )

        with self.assertRaises(ValueError):
            profile.validate()


if __name__ == "__main__":
    unittest.main()
