from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .records import ScoreRecord


def summarize(records: Iterable[ScoreRecord], by_subset: bool) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[ScoreRecord]] = defaultdict(list)
    for record in records:
        subset = record.subset if by_subset else "overall"
        groups[(record.model, record.dataset, subset)].append(record)

    rows = []
    for (model, dataset, subset), group in sorted(groups.items()):
        deltas = [record.delta for record in group]
        k = len(deltas)
        wins = sum(delta > 0 for delta in deltas)
        ties = sum(delta == 0 for delta in deltas)
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "subset": subset,
                "K": k,
                "A_hat": wins / k if k else float("nan"),
                "mu_hat": sum(deltas) / k if k else float("nan"),
                "tie_rate": ties / k if k else float("nan"),
                "mean_chosen_len": sum(r.chosen_token_count for r in group) / k if k else float("nan"),
                "mean_rejected_len": sum(r.rejected_token_count for r in group) / k if k else float("nan"),
            }
        )
    return rows


def rows_to_score_records(rows: list[dict[str, object]]) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    for row in rows:
        records.append(
            ScoreRecord(
                pair_id=str(row["pair_id"]),
                subset=str(row["subset"]),
                model=str(row["model"]),
                dataset=str(row["dataset"]),
                score_chosen=float(row["score_chosen"]),
                score_rejected=float(row["score_rejected"]),
                chosen_token_count=int(row["chosen_token_count"]),
                rejected_token_count=int(row["rejected_token_count"]),
                scoring_rule=str(row["scoring_rule"]),
            )
        )
    return records
