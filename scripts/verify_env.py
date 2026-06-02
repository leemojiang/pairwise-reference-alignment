from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pairalign.verify import verify_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify imports, CUDA, GPU memory, and SwanLab logging.")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--swanlab-project", default="pairwise-reference-alignment")
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument("--no-swanlab", action="store_true")
    args = parser.parse_args()
    verify_env(
        output_dir=args.output_dir,
        swanlab_project=args.swanlab_project,
        swanlab_workspace=args.swanlab_workspace,
        swanlab_enabled=not args.no_swanlab,
    )


if __name__ == "__main__":
    main()
