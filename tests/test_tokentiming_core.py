import unittest

from tokentiming import TokenTimingConfig, acceptance_probability
from tokentiming.result import GenerationStats


class TokenTimingCoreTest(unittest.TestCase):
    def test_config_validation_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            TokenTimingConfig(max_new_tokens=0).validate()
        with self.assertRaises(ValueError):
            TokenTimingConfig(num_draft_tokens=0).validate()
        with self.assertRaises(ValueError):
            TokenTimingConfig(dtw_window=-1).validate()

    def test_config_device_aliases(self):
        config = TokenTimingConfig(device="cpu")
        self.assertEqual(config.effective_target_device, "cpu")
        self.assertEqual(config.effective_draft_device, "cpu")

        config = TokenTimingConfig(device="cpu", target_device="cuda:0", draft_device="cuda:1")
        self.assertEqual(config.effective_target_device, "cuda:0")
        self.assertEqual(config.effective_draft_device, "cuda:1")

    def test_generation_stats_rates(self):
        stats = GenerationStats(
            generated_tokens=12,
            target_forwards=4,
            proposed_proxy_tokens=10,
            accepted_proxy_tokens=7,
        )

        self.assertEqual(stats.tokens_per_target_forward, 3.0)
        self.assertEqual(stats.acceptance_rate, 0.7)

    def test_acceptance_probability(self):
        self.assertEqual(acceptance_probability(0.9, 0.3), 1.0)
        self.assertEqual(acceptance_probability(0.25, 0.5), 0.5)
        with self.assertRaises(ValueError):
            acceptance_probability(-0.1, 0.5)


if __name__ == "__main__":
    unittest.main()
