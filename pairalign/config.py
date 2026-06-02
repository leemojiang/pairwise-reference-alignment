from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SwanLabConfig:
    enabled: bool = False
    project: str = "pairwise-reference-alignment"
    workspace: str | None = None


@dataclass(frozen=True)
class RunConfig:
    name: str = "debug"
    limit: int | None = None
    sample_size: int | None = None
    sample_by_subset: bool = False
    batch_size: int = 1
    max_length: int = 2048
    dtype: str = "auto"
    device: str = "auto"
    seed: int = 0
    trust_remote_code: bool = True
    resume: bool = True
    prompt_format: str = "plain"


@dataclass(frozen=True)
class ExperimentConfig:
    cache_dir: Path
    output_dir: Path
    models: list[str]
    datasets: list[str]
    scoring_rules: list[str] = field(default_factory=lambda: ["token_normalized_loglikelihood"])
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)
    run: RunConfig = field(default_factory=RunConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    base_dir = config_path.parent
    cache_dir = _path_from_config(data.get("cache_dir", "./cache"), base_dir)
    output_dir = _path_from_config(data.get("output_dir", "./runs"), base_dir)

    return ExperimentConfig(
        cache_dir=cache_dir,
        output_dir=output_dir,
        models=list(data.get("models", [])),
        datasets=list(data.get("datasets", [])),
        scoring_rules=list(data.get("scoring_rules", ["token_normalized_loglikelihood"])),
        swanlab=SwanLabConfig(**(data.get("swanlab") or {})),
        run=RunConfig(**(data.get("run") or {})),
    )


def _path_from_config(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path
