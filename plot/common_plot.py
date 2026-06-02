from __future__ import annotations

from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CODE_DIR.parent
DATA_ROOT = CODE_DIR / "remote_results" / "pairwise-reference-alignment-code"
OUTPUT_DIR = PROJECT_DIR / "Draft" / "Experiments-Result" / "plots"

PLAIN_RUN = DATA_ROOT / "runs" / "20260528-092001_experiment1and2_qwen_rewardbench_experiment12"
CHAT_RUN = DATA_ROOT / "runs" / "20260528-102610_experiment1and2_qwen_rewardbench_chat_experiment12"
FINITE_SAMPLE_RUN = DATA_ROOT / "runs" / "20260528-114143_experiment4_qwen_rewardbench_finite_sample"
BOOTSTRAP_RUN = DATA_ROOT / "runs" / "20260528-114147_experiment4b_qwen_rewardbench_bootstrap"


MODEL_ORDER = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
]


def short_model(model: str) -> str:
    return model.split("/")[-1].replace("Qwen2.5-", "")


def model_size(model: str) -> str:
    return short_model(model).replace("-Instruct", "")


def model_kind(model: str) -> str:
    return "Instruct" if model.endswith("-Instruct") else "Base"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
