from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pairalign.finite_sample import load_bootstrap_config, run_bootstrap_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 4B bootstrap confidence interval analysis.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_bootstrap_config(args.config)
    run_dir = run_bootstrap_experiment(config, args.config)
    print(f"Experiment 4B finished: {run_dir}")


if __name__ == "__main__":
    main()
