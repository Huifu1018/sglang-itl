import unittest


from tokentiming.alignment import dynamic_token_warping, levenshtein_distance
from tokentiming.prob_mapping import map_top1_draft_probabilities


class TokenTimingAlignmentTest(unittest.TestCase):
    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("abc", "abc"), 0)

    def test_dynamic_token_warping_exact_match(self):
        alignment = dynamic_token_warping(["hello", " world"], ["hello", " world"], window=1)

        self.assertEqual(alignment.total_cost, 0)
        self.assertEqual(alignment.target_to_draft, ((0,), (1,)))
        self.assertEqual(alignment.draft_to_target, ((0,), (1,)))

    def test_many_draft_tokens_to_one_target_token_uses_terminal_probability(self):
        alignment = dynamic_token_warping(["a", "b"], ["ab"])
        mapped = map_top1_draft_probabilities(
            draft_token_ids=[10, 11],
            target_token_ids=[20],
            alignment=alignment,
            draft_token_probabilities=[0.6, 0.9],
        )

        self.assertEqual(mapped[0].draft_indices, (0, 1))
        self.assertEqual(mapped[0].source_draft_index, 1)
        self.assertEqual(mapped[0].source_draft_token_id, 11)
        self.assertEqual(mapped[0].probability, 0.9)

    def test_one_draft_token_to_many_target_tokens_reuses_probability(self):
        alignment = dynamic_token_warping(["ab"], ["a", "b"])
        mapped = map_top1_draft_probabilities(
            draft_token_ids=[10],
            target_token_ids=[20, 21],
            alignment=alignment,
            draft_token_probabilities=[0.7],
        )

        self.assertEqual([item.draft_indices for item in mapped], [(0,), (0,)])
        self.assertEqual([item.source_draft_index for item in mapped], [0, 0])
        self.assertEqual([item.probability for item in mapped], [0.7, 0.7])


if __name__ == "__main__":
    unittest.main()
