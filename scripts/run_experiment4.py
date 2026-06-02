from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pairalign.finite_sample import load_finite_sample_config, run_finite_sample_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 4 finite-sample behavior analysis.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_finite_sample_config(args.config)
    run_dir = run_finite_sample_experiment(config, args.config)
    print(f"Experiment 4 finished: {run_dir}")


if __name__ == "__main__":
    main()
