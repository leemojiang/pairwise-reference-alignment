from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pairalign.smoke import run_smoke_cpu


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU-only smoke test with toy pairs and a mock scorer.")
    parser.add_argument("--config", default="configs/smoke_cpu.yaml")
    args = parser.parse_args()
    run_dir = run_smoke_cpu(args.config)
    print(f"CPU smoke finished: {run_dir}")


if __name__ == "__main__":
    main()
