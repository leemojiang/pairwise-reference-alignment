from __future__ import annotations

import unittest

from pairalign.datasets import dataset_label, normalize_pair


class RecordsTest(unittest.TestCase):
    def test_normalize_plain_pair(self) -> None:
        pair = normalize_pair(
            {
                "id": "x1",
                "prompt": "Question?",
                "chosen": "Good answer.",
                "rejected": "Bad answer.",
                "category": "helpfulness",
            },
            idx=0,
        )

        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(pair.pair_id, "x1")
        self.assertEqual(pair.subset, "helpfulness")
        self.assertEqual(pair.prompt, "Question?")
        self.assertEqual(pair.chosen, "Good answer.")
        self.assertEqual(pair.rejected, "Bad answer.")

    def test_normalize_message_pair(self) -> None:
        pair = normalize_pair(
            {
                "chosen": [
                    {"role": "user", "content": "Question?"},
                    {"role": "assistant", "content": "Good answer."},
                ],
                "rejected": [
                    {"role": "user", "content": "Question?"},
                    {"role": "assistant", "content": "Bad answer."},
                ],
            },
            idx=7,
        )

        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(pair.pair_id, "pair_000007")
        self.assertEqual(pair.prompt, "Question?")
        self.assertEqual(pair.chosen, "Good answer.")
        self.assertEqual(pair.rejected, "Bad answer.")

    def test_normalize_hhrlhf_transcript_pair(self) -> None:
        pair = normalize_pair(
            {
                "chosen": "\n\nHuman: How can I be safer online?\n\nAssistant: Use strong passwords and enable two-factor authentication.",
                "rejected": "\n\nHuman: How can I be safer online?\n\nAssistant: Share your passwords with friends so they can help.",
            },
            idx=3,
            dataset_id="Anthropic/hh-rlhf:helpful-base:train",
        )

        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertIn("Human: How can I be safer online?", pair.prompt)
        self.assertEqual(pair.chosen, "Use strong passwords and enable two-factor authentication.")
        self.assertEqual(pair.rejected, "Share your passwords with friends so they can help.")
        self.assertEqual(pair.subset, "Anthropic/hh-rlhf:helpful-base:train")

    def test_dataset_label_supports_string_and_dict(self) -> None:
        self.assertEqual(dataset_label("allenai/reward-bench"), "allenai/reward-bench")
        self.assertEqual(
            dataset_label({"name": "Anthropic/hh-rlhf", "subset": "helpful-base", "split": "train"}),
            "Anthropic/hh-rlhf:helpful-base:train",
        )


if __name__ == "__main__":
    unittest.main()
