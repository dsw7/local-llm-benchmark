#!/usr/bin/env python3

import logging
import sys
import core
from core.reporting import generate_report
from core.summary import print_summary
from core import run_benchmarks, BenchmarkError

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def main() -> None:
    try:
        configs = core.check_and_load_config()
    except BenchmarkError as e:
        sys.exit(str(e))

    try:
        run_benchmarks(configs)
    except BenchmarkError as e:
        sys.exit(str(e))

    print_summary(stats)
    generate_report(stats)


if __name__ == "__main__":
    main()
