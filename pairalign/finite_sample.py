from __future__ import annotations

import csv
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .io import make_run_dir


@dataclass(frozen=True)
class FiniteSampleSwanLabConfig:
    enabled: bool = False
    project: str = "pairwise-reference-alignment"
    workspace: str | None = None


@dataclass(frozen=True)
class FiniteSampleConfig:
    raw_scores_path: Path
    output_dir: Path
    run_name: str = "experiment4_finite_sample"
    k_values: list[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000, 2000])
    repeats: int = 200
    delta: float = 0.05
    seed: int = 20260527
    subset_enabled: bool = True
    subset_min_full_k: int = 50
    swanlab: FiniteSampleSwanLabConfig = field(default_factory=FiniteSampleSwanLabConfig)


@dataclass(frozen=True)
class BootstrapConfig:
    raw_scores_path: Path
    output_dir: Path
    run_name: str = "experiment4b_bootstrap_ci"
    repeats: int = 1000
    seed: int = 20260527
    subset_enabled: bool = True
    subset_min_full_k: int = 20
    swanlab: FiniteSampleSwanLabConfig = field(default_factory=FiniteSampleSwanLabConfig)


@dataclass(frozen=True)
class FiniteSampleRecord:
    pair_id: str
    model: str
    dataset: str
    subset: str
    scoring_rule: str
    win: int
    delta: float


def load_finite_sample_config(path: str | Path) -> FiniteSampleConfig:
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    base_dir = config_path.parent
    raw_scores_path = _path_from_config(data["raw_scores_path"], base_dir)
    output_dir = _path_from_config(data.get("output_dir", "../runs"), base_dir)
    return FiniteSampleConfig(
        raw_scores_path=raw_scores_path,
        output_dir=output_dir,
        run_name=str(data.get("run_name", "experiment4_finite_sample")),
        k_values=[int(k) for k in data.get("k_values", [50, 100, 200, 500, 1000, 2000])],
        repeats=int(data.get("repeats", 200)),
        delta=float(data.get("delta", 0.05)),
        seed=int(data.get("seed", 20260527)),
        subset_enabled=bool(data.get("subset_enabled", True)),
        subset_min_full_k=int(data.get("subset_min_full_k", 50)),
        swanlab=FiniteSampleSwanLabConfig(**(data.get("swanlab") or {})),
    )


def load_bootstrap_config(path: str | Path) -> BootstrapConfig:
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    base_dir = config_path.parent
    raw_scores_path = _path_from_config(data["raw_scores_path"], base_dir)
    output_dir = _path_from_config(data.get("output_dir", "../runs"), base_dir)
    return BootstrapConfig(
        raw_scores_path=raw_scores_path,
        output_dir=output_dir,
        run_name=str(data.get("run_name", "experiment4b_bootstrap_ci")),
        repeats=int(data.get("repeats", 1000)),
        seed=int(data.get("seed", 20260527)),
        subset_enabled=bool(data.get("subset_enabled", True)),
        subset_min_full_k=int(data.get("subset_min_full_k", 20)),
        swanlab=FiniteSampleSwanLabConfig(**(data.get("swanlab") or {})),
    )


def run_finite_sample_experiment(config: FiniteSampleConfig, config_path: str | Path) -> Path:
    records = read_raw_score_records(config.raw_scores_path)
    run_dir = make_run_dir(config.output_dir, config.run_name)
    shutil.copy2(config_path, run_dir / "config.yaml")

    overall_trials, overall_summary = analyze_finite_sample(
        records,
        k_values=config.k_values,
        repeats=config.repeats,
        delta=config.delta,
        seed=config.seed,
        group_fields=("model", "dataset", "scoring_rule"),
    )
    write_csv(run_dir / "finite_sample_overall_trials.csv", overall_trials)
    write_csv(run_dir / "finite_sample_overall_summary.csv", overall_summary)

    subset_summary: list[dict[str, object]] = []
    if config.subset_enabled:
        _, subset_summary = analyze_finite_sample(
            records,
            k_values=config.k_values,
            repeats=config.repeats,
            delta=config.delta,
            seed=config.seed,
            group_fields=("model", "dataset", "scoring_rule", "subset"),
            min_full_k=config.subset_min_full_k,
        )
        write_csv(run_dir / "finite_sample_subset_summary.csv", subset_summary)

    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "raw_scores_path": str(config.raw_scores_path),
                "records": len(records),
                "k_values": config.k_values,
                "repeats": config.repeats,
                "delta": config.delta,
                "seed": config.seed,
                "subset_enabled": config.subset_enabled,
                "subset_min_full_k": config.subset_min_full_k,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def run_bootstrap_experiment(config: BootstrapConfig, config_path: str | Path) -> Path:
    records = read_raw_score_records(config.raw_scores_path)
    run_dir = make_run_dir(config.output_dir, config.run_name)
    shutil.copy2(config_path, run_dir / "config.yaml")

    overall_trials, overall_ci = analyze_bootstrap(
        records,
        repeats=config.repeats,
        seed=config.seed,
        group_fields=("model", "dataset", "scoring_rule"),
    )
    write_csv(run_dir / "bootstrap_overall_trials.csv", overall_trials)
    write_csv(run_dir / "bootstrap_overall_ci.csv", overall_ci)

    if config.subset_enabled:
        _, subset_ci = analyze_bootstrap(
            records,
            repeats=config.repeats,
            seed=config.seed,
            group_fields=("model", "dataset", "scoring_rule", "subset"),
            min_full_k=config.subset_min_full_k,
        )
        write_csv(run_dir / "bootstrap_subset_ci.csv", subset_ci)

    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "raw_scores_path": str(config.raw_scores_path),
                "records": len(records),
                "repeats": config.repeats,
                "seed": config.seed,
                "subset_enabled": config.subset_enabled,
                "subset_min_full_k": config.subset_min_full_k,
                "bootstrap_sample_size": "full group size for each group",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def read_raw_score_records(path: str | Path) -> list[FiniteSampleRecord]:
    records: list[FiniteSampleRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            delta = float(row["delta"])
            records.append(
                FiniteSampleRecord(
                    pair_id=str(row["pair_id"]),
                    model=str(row["model"]),
                    dataset=str(row["dataset"]),
                    subset=str(row["subset"]),
                    scoring_rule=str(row["scoring_rule"]),
                    win=int(delta > 0),
                    delta=delta,
                )
            )
    return records


def analyze_finite_sample(
    records: Iterable[FiniteSampleRecord],
    *,
    k_values: list[int],
    repeats: int,
    delta: float,
    seed: int,
    group_fields: tuple[str, ...],
    min_full_k: int = 1,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, ...], list[FiniteSampleRecord]] = defaultdict(list)
    for record in records:
        grouped[tuple(str(getattr(record, field_name)) for field_name in group_fields)].append(record)

    trials: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for group_key, group in sorted(grouped.items()):
        full_k = len(group)
        if full_k < min_full_k:
            continue
        full_a_hat = sum(r.win for r in group) / full_k
        full_mu_hat = sum(r.delta for r in group) / full_k
        group_base = dict(zip(group_fields, group_key))
        group_seed = _stable_group_seed(seed, group_key)
        rng = random.Random(group_seed)

        for k in sorted(k for k in k_values if k <= full_k):
            a_values = []
            mu_values = []
            for repeat_index in range(repeats):
                sampled = rng.sample(group, k)
                a_hat = sum(r.win for r in sampled) / k
                mu_hat = sum(r.delta for r in sampled) / k
                a_values.append(a_hat)
                mu_values.append(mu_hat)
                trials.append(
                    group_base
                    | {
                        "K": k,
                        "repeat": repeat_index,
                        "A_hat": a_hat,
                        "mu_hat": mu_hat,
                        "full_K": full_k,
                        "full_A_hat": full_a_hat,
                        "full_mu_hat": full_mu_hat,
                    }
                )
            summary.append(group_base | summarize_values(a_values, mu_values, k, full_k, full_a_hat, full_mu_hat, delta, repeats))
    return trials, summary


def analyze_bootstrap(
    records: Iterable[FiniteSampleRecord],
    *,
    repeats: int,
    seed: int,
    group_fields: tuple[str, ...],
    min_full_k: int = 1,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, ...], list[FiniteSampleRecord]] = defaultdict(list)
    for record in records:
        grouped[tuple(str(getattr(record, field_name)) for field_name in group_fields)].append(record)

    trials: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for group_key, group in sorted(grouped.items()):
        full_k = len(group)
        if full_k < min_full_k:
            continue
        full_a_hat = sum(r.win for r in group) / full_k
        full_mu_hat = sum(r.delta for r in group) / full_k
        group_base = dict(zip(group_fields, group_key))
        group_seed = _stable_group_seed(seed, group_key + ("bootstrap",))
        rng = random.Random(group_seed)
        a_values = []
        mu_values = []
        for repeat_index in range(repeats):
            sampled = rng.choices(group, k=full_k)
            a_hat = sum(r.win for r in sampled) / full_k
            mu_hat = sum(r.delta for r in sampled) / full_k
            a_values.append(a_hat)
            mu_values.append(mu_hat)
            trials.append(
                group_base
                | {
                    "K": full_k,
                    "repeat": repeat_index,
                    "A_hat": a_hat,
                    "mu_hat": mu_hat,
                    "full_K": full_k,
                    "full_A_hat": full_a_hat,
                    "full_mu_hat": full_mu_hat,
                    "sampling": "with_replacement",
                }
            )
        summary.append(group_base | summarize_bootstrap_values(a_values, mu_values, full_k, full_a_hat, full_mu_hat, repeats))
    return trials, summary


def summarize_values(
    a_values: list[float],
    mu_values: list[float],
    k: int,
    full_k: int,
    full_a_hat: float,
    full_mu_hat: float,
    delta: float,
    repeats: int,
) -> dict[str, object]:
    mean = sum(a_values) / len(a_values)
    sorted_values = sorted(a_values)
    sd = _sample_sd(a_values)
    ci_low = _quantile(sorted_values, 0.025)
    ci_high = _quantile(sorted_values, 0.975)
    abs_errors = [abs(value - full_a_hat) for value in a_values]
    empirical_abs_error_p95 = _quantile(sorted(abs_errors), 0.95)
    hoeffding_radius = math.sqrt(math.log(2 / delta) / (2 * k))
    coverage = sum(error <= hoeffding_radius for error in abs_errors) / len(abs_errors)
    mu_summary = _summarize_metric(mu_values, full_mu_hat)
    return {
        "K": k,
        "full_K": full_k,
        "full_A_hat": full_a_hat,
        "full_mu_hat": full_mu_hat,
        "repeats": repeats,
        "mean_A_hat": mean,
        "sd_A_hat": sd,
        "ci_low_2p5": ci_low,
        "ci_high_97p5": ci_high,
        "ci_width_95": ci_high - ci_low,
        "empirical_abs_error_p95": empirical_abs_error_p95,
        "hoeffding_radius_delta": delta,
        "hoeffding_radius": hoeffding_radius,
        "hoeffding_coverage_vs_full": coverage,
        "mean_mu_hat": mu_summary["mean"],
        "sd_mu_hat": mu_summary["sd"],
        "mu_ci_low_2p5": mu_summary["ci_low"],
        "mu_ci_high_97p5": mu_summary["ci_high"],
        "mu_ci_width_95": mu_summary["ci_width"],
        "mu_empirical_abs_error_p95": mu_summary["abs_error_p95"],
    }


def summarize_bootstrap_values(
    a_values: list[float],
    mu_values: list[float],
    full_k: int,
    full_a_hat: float,
    full_mu_hat: float,
    repeats: int,
) -> dict[str, object]:
    sorted_values = sorted(a_values)
    ci_low = _quantile(sorted_values, 0.025)
    ci_high = _quantile(sorted_values, 0.975)
    mean = sum(a_values) / len(a_values)
    mu_summary = _summarize_metric(mu_values, full_mu_hat)
    return {
        "K": full_k,
        "full_K": full_k,
        "full_A_hat": full_a_hat,
        "full_mu_hat": full_mu_hat,
        "repeats": repeats,
        "bootstrap_mean_A_hat": mean,
        "bootstrap_bias": mean - full_a_hat,
        "bootstrap_sd_A_hat": _sample_sd(a_values),
        "bootstrap_ci_low_2p5": ci_low,
        "bootstrap_ci_high_97p5": ci_high,
        "bootstrap_ci_width_95": ci_high - ci_low,
        "bootstrap_mean_mu_hat": mu_summary["mean"],
        "bootstrap_mu_bias": mu_summary["mean"] - full_mu_hat,
        "bootstrap_sd_mu_hat": mu_summary["sd"],
        "bootstrap_mu_ci_low_2p5": mu_summary["ci_low"],
        "bootstrap_mu_ci_high_97p5": mu_summary["ci_high"],
        "bootstrap_mu_ci_width_95": mu_summary["ci_width"],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[lower]
    weight = pos - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _summarize_metric(values: list[float], full_value: float) -> dict[str, float]:
    sorted_values = sorted(values)
    ci_low = _quantile(sorted_values, 0.025)
    ci_high = _quantile(sorted_values, 0.975)
    abs_errors = [abs(value - full_value) for value in values]
    return {
        "mean": sum(values) / len(values),
        "sd": _sample_sd(values),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
        "abs_error_p95": _quantile(sorted(abs_errors), 0.95),
    }


def _stable_group_seed(seed: int, group_key: tuple[str, ...]) -> int:
    value = seed
    for part in group_key:
        for char in part:
            value = (value * 131 + ord(char)) % (2**32)
    return value


def _path_from_config(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path
