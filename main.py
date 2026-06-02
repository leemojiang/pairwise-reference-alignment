from __future__ import annotations

import argparse

from pairalign.config import load_config
from pairalign.run import run_experiment
from pairalign.smoke import run_smoke_cpu
from pairalign.verify import prepare_cache, tune_batch_size, verify_datasets, verify_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise reference alignment helper CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    env_parser = subparsers.add_parser("verify-env")
    env_parser.add_argument("--output-dir", default="runs")
    env_parser.add_argument("--swanlab-project", default="pairwise-reference-alignment")
    env_parser.add_argument("--swanlab-workspace", default=None)
    env_parser.add_argument("--no-swanlab", action="store_true")

    for command in ["verify-datasets", "tune-batch", "experiment1", "experiment2", "experiment1and2"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", default="configs/smoke.yaml")
    subparsers.choices["experiment1and2"].set_defaults(config="configs/experiment1and2_qwen_rewardbench.yaml")
    prepare_parser = subparsers.add_parser("prepare-cache")
    prepare_parser.add_argument("--config", default="configs/prepare_cache.yaml")
    smoke_parser = subparsers.add_parser("smoke-cpu")
    smoke_parser.add_argument("--config", default="configs/smoke_cpu.yaml")
    subparsers.choices["tune-batch"].add_argument("--batch-sizes", default="1,2,4,8")

    args = parser.parse_args()
    if args.command == "verify-env":
        verify_env(
            output_dir=args.output_dir,
            swanlab_project=args.swanlab_project,
            swanlab_workspace=args.swanlab_workspace,
            swanlab_enabled=not args.no_swanlab,
        )
        return

    if args.command == "smoke-cpu":
        run_smoke_cpu(args.config)
        return

    config = load_config(args.config)
    if args.command == "verify-datasets":
        verify_datasets(config)
    elif args.command == "prepare-cache":
        prepare_cache(args.config)
    elif args.command == "tune-batch":
        tune_batch_size(config, [int(item) for item in args.batch_sizes.split(",") if item.strip()])
    elif args.command == "experiment1":
        run_experiment(config, args.config, experiment_id=1)
    elif args.command == "experiment2":
        run_experiment(config, args.config, experiment_id=2)
    elif args.command == "experiment1and2":
        run_experiment(config, args.config, experiment_id=12)


if __name__ == "__main__":
    main()
