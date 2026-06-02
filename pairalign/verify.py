from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import ExperimentConfig
from .datasets import dataset_label, load_pair_dataset
from .io import append_jsonl, make_run_dir
from .scoring import HFScorer
from .swanlab_utils import SwanLabRun


REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("huggingface_hub", "huggingface-hub"),
    ("numpy", "numpy"),
    ("yaml", "PyYAML"),
    ("swanlab", "swanlab"),
]


def verify_env(
    output_dir: str | Path = "runs",
    swanlab_project: str = "pairwise-reference-alignment",
    swanlab_workspace: str | None = None,
    swanlab_enabled: bool = True,
) -> Path:
    run_dir = make_run_dir(Path(output_dir), "ENV_check")
    package_report = {
        import_name: {
            "import_ok": _package_available(import_name),
            "version": _package_version(distribution_name),
        }
        for import_name, distribution_name in REQUIRED_PACKAGES
    }
    report = {
        "check_name": "ENV check",
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [_cuda_device_info(i) for i in range(torch.cuda.device_count())],
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "SWANLAB_API_KEY_SET": bool(os.environ.get("SWANLAB_API_KEY")),
        },
        "packages": package_report,
    }
    swan = SwanLabRun(
        swanlab_enabled,
        swanlab_project,
        swanlab_workspace,
        "ENV check",
        report,
    )
    report["swanlab_enabled"] = swanlab_enabled
    report["swanlab_init_ok"] = swan.init_ok
    report["swanlab_init_error"] = swan.init_error
    (run_dir / "verify_env.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    swan.log({"verify/cuda_available": report["cuda_available"], "verify/cuda_device_count": report["cuda_device_count"]})
    swan.log({"verify/swanlab_init_ok": report["swanlab_init_ok"]})
    for name, item in report["packages"].items():
        swan.log({f"verify/package_{name}": item["import_ok"]})
    swan.finish()
    print(json.dumps(report, indent=2))
    return run_dir


def verify_datasets(config: ExperimentConfig) -> Path:
    run_dir = make_run_dir(config.output_dir, f"{config.run.name}_verify_datasets")
    rows = []
    swan = SwanLabRun(config.swanlab.enabled, config.swanlab.project, config.swanlab.workspace, f"{config.run.name}-verify-datasets", config)
    for dataset_spec in config.datasets:
        dataset_name = dataset_label(dataset_spec)
        try:
            records = load_pair_dataset(dataset_spec, config.cache_dir, limit=config.run.limit or 20)
            row = {"dataset": dataset_name, "ok": True, "loaded_pairs": len(records)}
            if not records:
                row = {"dataset": dataset_name, "ok": False, "loaded_pairs": 0, "error": "No pair records were normalized."}
        except Exception as exc:
            row = {"dataset": dataset_name, "ok": False, "error": str(exc)}
        rows.append(row)
        swan.log({f"dataset/{dataset_name}/ok": int(row["ok"]), f"dataset/{dataset_name}/loaded_pairs": row.get("loaded_pairs", 0)})
    append_jsonl(run_dir / "verify_datasets.jsonl", rows)
    swan.finish()
    for row in rows:
        print(row)
    return run_dir


def prepare_cache(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    cache_dir = Path(config.get("cache_dir", "./cache")).expanduser().resolve()
    output_dir = Path(config.get("output_dir", "./runs")).expanduser().resolve()
    hf_endpoint = config.get("hf_endpoint")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = str(hf_endpoint)
    os.environ["HF_HOME"] = str(cache_dir / "hf_home")
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")

    cache_dir.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(output_dir, "CACHE_check")
    swanlab_config = config.get("swanlab") or {}
    swan = SwanLabRun(
        bool(swanlab_config.get("enabled", True)),
        str(swanlab_config.get("project", "pairwise-reference-alignment")),
        swanlab_config.get("workspace"),
        "CACHE check",
        config,
    )

    report = {
        "check_name": "CACHE check",
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "cache_dir": str(cache_dir),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
        "hf_token_available": _hf_token_available(),
        "swanlab_init_ok": swan.init_ok,
        "swanlab_init_error": swan.init_error,
    }
    (run_dir / "cache_env.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    swan.log({"cache/hf_token_available": report["hf_token_available"], "cache/swanlab_init_ok": report["swanlab_init_ok"]})

    download_config = config.get("download") or {}
    huggingface_config = config.get("huggingface") or {}
    token_env = huggingface_config.get("token_env", "HF_TOKEN")
    token = os.environ.get(token_env) if token_env else None
    trust_remote_code = bool(download_config.get("trust_remote_code", True))
    download_datasets = bool(download_config.get("download_datasets", True))
    download_models = bool(download_config.get("download_models", True))
    dataset_limit = download_config.get("dataset_verify_limit", 20)

    rows = []
    if download_datasets:
        for dataset_spec in config.get("datasets", []):
            dataset_name = dataset_label(dataset_spec)
            started = time.perf_counter()
            try:
                records = load_pair_dataset(dataset_spec, cache_dir, limit=dataset_limit)
                row = {
                    "kind": "dataset",
                    "name": dataset_name,
                    "ok": bool(records),
                    "loaded_pairs": len(records),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                if not records:
                    row["error"] = "No pair records were normalized."
            except Exception as exc:
                row = {
                    "kind": "dataset",
                    "name": dataset_name,
                    "ok": False,
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            rows.append(row)
            append_jsonl(run_dir / "cache_items.jsonl", [row])
            swan.log({f"cache/dataset/{dataset_name}/ok": int(row["ok"]), f"cache/dataset/{dataset_name}/elapsed_seconds": row["elapsed_seconds"]})
            print(row)

    if download_models:
        from huggingface_hub import snapshot_download

        for model_name in config.get("models", []):
            started = time.perf_counter()
            try:
                local_path = snapshot_download(
                    repo_id=model_name,
                    cache_dir=str(cache_dir / "hub"),
                    local_files_only=False,
                    token=token,
                    endpoint=str(hf_endpoint) if hf_endpoint else None,
                    ignore_patterns=download_config.get("ignore_patterns"),
                    allow_patterns=download_config.get("allow_patterns"),
                )
                row = {
                    "kind": "model",
                    "name": model_name,
                    "ok": True,
                    "local_path": local_path,
                    "trust_remote_code": trust_remote_code,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            except Exception as exc:
                row = {
                    "kind": "model",
                    "name": model_name,
                    "ok": False,
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            rows.append(row)
            append_jsonl(run_dir / "cache_items.jsonl", [row])
            swan.log({f"cache/model/{model_name}/ok": int(row["ok"]), f"cache/model/{model_name}/elapsed_seconds": row["elapsed_seconds"]})
            print(row)

    ok_count = sum(1 for row in rows if row["ok"])
    swan.log({"cache/items_total": len(rows), "cache/items_ok": ok_count})
    swan.finish()
    return run_dir


def tune_batch_size(config: ExperimentConfig, batch_sizes: list[int]) -> Path:
    run_dir = make_run_dir(config.output_dir, f"{config.run.name}_tune_batch")
    swan = SwanLabRun(config.swanlab.enabled, config.swanlab.project, config.swanlab.workspace, f"{config.run.name}-tune-batch", config)
    rows = []
    dataset_spec = config.datasets[0]
    dataset_name = dataset_label(dataset_spec)
    records = load_pair_dataset(dataset_spec, config.cache_dir, limit=max(config.run.limit or 4, 4))
    for model_name in config.models:
        for batch_size in batch_sizes:
            run = config.run.__class__(**{**config.run.__dict__, "batch_size": batch_size, "limit": len(records)})
            scorer = None
            try:
                scorer = HFScorer(model_name, config.cache_dir, run)
                _, timings = scorer.score_pairs(records, dataset_name, config.scoring_rules[0])
                ok = True
                error = ""
            except RuntimeError as exc:
                ok = False
                timings = []
                error = str(exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                if scorer is not None:
                    del scorer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row = {"model": model_name, "batch_size": batch_size, "ok": ok, "error": error}
            rows.append(row)
            append_jsonl(run_dir / "batch_trials.jsonl", [row])
            append_jsonl(run_dir / "batch_timings.jsonl", [t.to_json() | {"model": model_name, "batch_size": batch_size} for t in timings])
            swan.log({f"batch/{model_name}/{batch_size}/ok": int(ok)})
            for timing in timings:
                swan.log(
                    {
                        "batch/elapsed_seconds": timing.elapsed_seconds,
                        "batch/tokens_per_second": timing.tokens_per_second,
                        "batch/batch_size": batch_size,
                    },
                    step=timing.batch_index,
                )
    swan.finish()
    return run_dir


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cuda_device_info(index: int) -> dict[str, object]:
    props = torch.cuda.get_device_properties(index)
    return {
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "major": props.major,
        "minor": props.minor,
    }


def _hf_token_available() -> bool:
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token() is not None
    except Exception:
        return False
