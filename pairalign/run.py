from __future__ import annotations

import json
import csv
import random
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .datasets import dataset_label, load_pair_dataset
from .io import append_jsonl, copy_config, make_run_dir, read_done_keys, score_records_to_rows
from .metrics import summarize
from .records import PairRecord
from .scoring import HFScorer
from .swanlab_utils import SwanLabRun


def run_experiment(config: ExperimentConfig, config_path: str | Path, experiment_id: int) -> Path:
    _set_seed(config.run.seed)
    run_dir = make_run_dir(config.output_dir, f"{config.run.name}_experiment{experiment_id}")
    copy_config(config_path, run_dir)
    raw_path = run_dir / "raw_scores.jsonl"
    timings_path = run_dir / "batch_timings.jsonl"
    swan = SwanLabRun(
        config.swanlab.enabled,
        config.swanlab.project,
        config.swanlab.workspace,
        f"{config.run.name}-experiment{experiment_id}",
        config,
    )

    all_records = []
    done_keys = read_done_keys(raw_path) if config.run.resume else set()
    for dataset_spec in config.datasets:
        dataset_name = dataset_label(dataset_spec)
        pairs = load_pair_dataset(dataset_spec, config.cache_dir, limit=config.run.limit)
        pairs = _sample_pairs(pairs, config.run.sample_size, config.run.sample_by_subset, config.run.seed)
        for model_name in config.models:
            scorer = HFScorer(model_name, config.cache_dir, config.run)
            try:
                for scoring_rule in config.scoring_rules:
                    pending = [
                        pair
                        for pair in pairs
                        if (dataset_name, model_name, scoring_rule, pair.pair_id) not in done_keys
                    ]
                    records, timings = scorer.score_pairs(pending, dataset_name, scoring_rule)
                    append_jsonl(raw_path, score_records_to_rows(records))
                    append_jsonl(
                        timings_path,
                        [
                            timing.to_json()
                            | {
                                "dataset": dataset_name,
                                "model": model_name,
                                "scoring_rule": scoring_rule,
                            }
                            for timing in timings
                        ],
                    )
                    all_records.extend(records)
                    for timing in timings:
                        swan.log(
                            {
                                "batch/elapsed_seconds": timing.elapsed_seconds,
                                "batch/tokens_per_second": timing.tokens_per_second,
                                "batch/pairs": timing.pairs,
                            },
                            step=timing.batch_index,
                        )
            finally:
                del scorer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    overall = summarize(all_records, by_subset=False)
    _write_csv(run_dir / "metrics_overall.csv", overall)
    _log_metrics(swan, overall, prefix="overall")

    if experiment_id in {2, 12}:
        by_subset = summarize(all_records, by_subset=True)
        _write_csv(run_dir / "metrics_by_subset.csv", by_subset)
        _log_metrics(swan, by_subset, prefix="subset")

    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "models": config.models,
                "datasets": config.datasets,
                "scoring_rules": config.scoring_rules,
                "limit": config.run.limit,
                "sample_size": config.run.sample_size,
                "sample_by_subset": config.run.sample_by_subset,
                "batch_size": config.run.batch_size,
                "max_length": config.run.max_length,
                "dtype": config.run.dtype,
                "device": config.run.device,
                "seed": config.run.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    swan.finish()
    return run_dir


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _log_metrics(swan: SwanLabRun, rows: list[dict[str, object]], prefix: str) -> None:
    for row in rows:
        key_base = f"{prefix}/{row['model']}/{row['dataset']}/{row['subset']}"
        swan.log(
            {
                f"{key_base}/K": int(row["K"]),
                f"{key_base}/A_hat": float(row["A_hat"]),
                f"{key_base}/mu_hat": float(row["mu_hat"]),
                f"{key_base}/tie_rate": float(row["tie_rate"]),
            }
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_pairs(pairs: list[PairRecord], sample_size: int | None, by_subset: bool, seed: int) -> list[PairRecord]:
    if sample_size is None or sample_size >= len(pairs):
        return pairs
    rng = random.Random(seed)
    if not by_subset:
        return rng.sample(pairs, sample_size)

    grouped: dict[str, list[PairRecord]] = {}
    for pair in pairs:
        grouped.setdefault(pair.subset, []).append(pair)

    subset_names = sorted(grouped)
    base_n = max(1, sample_size // len(subset_names))
    sampled: list[PairRecord] = []
    for subset_name in subset_names:
        group = grouped[subset_name]
        sampled.extend(rng.sample(group, min(base_n, len(group))))

    remaining = sample_size - len(sampled)
    if remaining > 0:
        selected_ids = {pair.pair_id for pair in sampled}
        pool = [pair for pair in pairs if pair.pair_id not in selected_ids]
        sampled.extend(rng.sample(pool, min(remaining, len(pool))))
    return sampled
