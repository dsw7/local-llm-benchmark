#!/usr/bin/env python3

import argparse
import logging
import sys
import core

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark locally hosted LLMs")

    parser.add_argument(
        "-p",
        "--generate-plots",
        action="store_true",
        help="Generate normal distribution plots",
    )

    args = parser.parse_args()
    return args


def run_subprogram(args: argparse.Namespace, configs: core.models.Configs) -> None:
    if args.generate_plots:
        core.export_normal_distribution_plots()
        return

    core.run_benchmarks(configs)


def main() -> None:
    args = parse_args()

    try:
        configs = core.check_and_load_config()
    except core.ConfigError as e:
        sys.exit(str(e))

    try:
        run_subprogram(args, configs)
    except core.BenchmarkError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
