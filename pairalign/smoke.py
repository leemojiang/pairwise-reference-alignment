from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from .datasets import load_pair_dataset
from .io import append_jsonl, copy_config, make_run_dir, score_records_to_rows
from .metrics import summarize
from .records import PairRecord, ScoreRecord
from .run import _log_metrics, _write_csv
from .swanlab_utils import SwanLabRun


def run_smoke_cpu(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    seed = int(config.get("seed", 0))
    random.seed(seed)
    output_dir = Path(config.get("output_dir", "./runs")).expanduser().resolve()
    cache_dir = Path(config.get("cache_dir", "./cache")).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(output_dir, "smoke_cpu")
    copy_config(config_path, run_dir)

    swanlab_config = config.get("swanlab") or {}
    swan = SwanLabRun(
        bool(swanlab_config.get("enabled", True)),
        str(swanlab_config.get("project", "pairwise-reference-alignment")),
        swanlab_config.get("workspace"),
        "SMOKE CPU",
        config,
    )

    dataset_path = config.get("dataset_path", "fixtures/toy_pairs.jsonl")
    pairs = load_pair_dataset(dataset_path, cache_dir, limit=config.get("limit"))
    records = [_mock_score(pair, config.get("model", "mock-cpu-model"), dataset_path) for pair in pairs]

    append_jsonl(run_dir / "raw_scores.jsonl", score_records_to_rows(records))
    overall = summarize(records, by_subset=False)
    by_subset = summarize(records, by_subset=True)
    _write_csv(run_dir / "metrics_overall.csv", overall)
    _write_csv(run_dir / "metrics_by_subset.csv", by_subset)
    _log_metrics(swan, overall, "overall")
    _log_metrics(swan, by_subset, "subset")

    meta = {
        "smoke_kind": "cpu_mock",
        "seed": seed,
        "cache_dir": str(cache_dir),
        "dataset_path": dataset_path,
        "pair_count": len(pairs),
        "swanlab_enabled": bool(swanlab_config.get("enabled", True)),
        "swanlab_init_ok": swan.init_ok,
        "swanlab_init_error": swan.init_error,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    swan.log({"smoke/pair_count": len(pairs), "smoke/seed": seed, "smoke/swanlab_init_ok": swan.init_ok})
    swan.finish()
    return run_dir


def _mock_score(pair: PairRecord, model: str, dataset: str) -> ScoreRecord:
    chosen_score = _text_score(pair.prompt, pair.chosen)
    rejected_score = _text_score(pair.prompt, pair.rejected)
    return ScoreRecord(
        pair_id=pair.pair_id,
        subset=pair.subset,
        model=model,
        dataset=dataset,
        score_chosen=chosen_score,
        score_rejected=rejected_score,
        chosen_token_count=max(1, len(pair.chosen.split())),
        rejected_token_count=max(1, len(pair.rejected.split())),
        scoring_rule="mock_token_normalized_loglikelihood",
    )


def _text_score(prompt: str, response: str) -> float:
    text = f"{prompt}\n{response}"
    checksum = sum(ord(char) for char in text)
    length_penalty = len(response.split()) * 0.001
    return (checksum % 1000) / 1000.0 - length_penalty

