from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pairalign.config import load_config
from pairalign.verify import verify_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Hugging Face datasets can be loaded and normalized.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    verify_datasets(load_config(args.config))


if __name__ == "__main__":
    main()
