#!/usr/bin/env python3

import logging
import sys
import core

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def main() -> None:
    try:
        configs = core.check_and_load_config()
    except core.BenchmarkError as e:
        sys.exit(str(e))

    try:
        stats = core.run_benchmarks(configs)
    except core.BenchmarkError as e:
        sys.exit(str(e))

    core.print_summary_to_stdout(stats)


if __name__ == "__main__":
    main()
