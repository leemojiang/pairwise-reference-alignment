from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pairalign.verify import prepare_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Hugging Face dataset/model cache before experiments.")
    parser.add_argument("--config", default="configs/prepare_cache.yaml")
    args = parser.parse_args()
    run_dir = prepare_cache(args.config)
    print(f"Cache check finished: {run_dir}")


if __name__ == "__main__":
    main()
