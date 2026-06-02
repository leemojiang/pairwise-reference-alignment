from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    subset: str = "overall"


@dataclass(frozen=True)
class ScoreRecord:
    pair_id: str
    subset: str
    model: str
    dataset: str
    score_chosen: float
    score_rejected: float
    chosen_token_count: int
    rejected_token_count: int
    scoring_rule: str

    @property
    def delta(self) -> float:
        return self.score_chosen - self.score_rejected

    def to_json(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "subset": self.subset,
            "model": self.model,
            "dataset": self.dataset,
            "score_chosen": self.score_chosen,
            "score_rejected": self.score_rejected,
            "delta": self.delta,
            "chosen_token_count": self.chosen_token_count,
            "rejected_token_count": self.rejected_token_count,
            "scoring_rule": self.scoring_rule,
        }

