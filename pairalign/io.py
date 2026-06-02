from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .records import ScoreRecord


def make_run_dir(output_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def copy_config(config_path: str | Path, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / "config.yaml")


def append_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_done_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((row["dataset"], row["model"], row["scoring_rule"], row["pair_id"]))
    return keys


def score_records_to_rows(records: Iterable[ScoreRecord]) -> list[dict[str, object]]:
    return [record.to_json() for record in records]

