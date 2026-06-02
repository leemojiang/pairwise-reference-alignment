from __future__ import annotations

import unittest

from pairalign.metrics import summarize
from pairalign.records import ScoreRecord


class MetricsTest(unittest.TestCase):
    def test_overall_alignment_uses_chosen_minus_rejected(self) -> None:
        records = [
            _record("a", "chat", 2.0, 1.0),
            _record("b", "chat", 1.0, 2.0),
            _record("c", "safety", 3.0, 1.0),
        ]

        frame = summarize(records, by_subset=False)
        row = frame[0]

        self.assertEqual(row["K"], 3)
        self.assertAlmostEqual(row["A_hat"], 2 / 3)
        self.assertAlmostEqual(row["mu_hat"], (1.0 - 1.0 + 2.0) / 3)

    def test_subset_alignment_groups_by_subset(self) -> None:
        records = [
            _record("a", "chat", 2.0, 1.0),
            _record("b", "chat", 1.0, 2.0),
            _record("c", "safety", 3.0, 1.0),
        ]

        frame = summarize(records, by_subset=True)
        rows = {row["subset"]: row for row in frame}

        self.assertAlmostEqual(rows["chat"]["A_hat"], 0.5)
        self.assertAlmostEqual(rows["safety"]["A_hat"], 1.0)


def _record(pair_id: str, subset: str, chosen: float, rejected: float) -> ScoreRecord:
    return ScoreRecord(
        pair_id=pair_id,
        subset=subset,
        model="mock-model",
        dataset="mock-dataset",
        score_chosen=chosen,
        score_rejected=rejected,
        chosen_token_count=10,
        rejected_token_count=12,
        scoring_rule="token_normalized_loglikelihood",
    )


if __name__ == "__main__":
    unittest.main()
