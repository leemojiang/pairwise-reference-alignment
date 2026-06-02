from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pairalign.config import load_config
from pairalign.run import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 1+2: overall and subset-level alignment.")
    parser.add_argument("--config", default="configs/experiment1and2_qwen_rewardbench.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    run_dir = run_experiment(config, config_path, experiment_id=12)
    print(f"Experiment 1+2 finished: {run_dir}")


if __name__ == "__main__":
    main()
