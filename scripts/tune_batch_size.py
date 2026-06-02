from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pairalign.config import load_config
from pairalign.verify import tune_batch_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Try candidate batch sizes and record batch timing.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    args = parser.parse_args()
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    tune_batch_size(load_config(args.config), batch_sizes)


if __name__ == "__main__":
    main()
