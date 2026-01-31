import logging
from tabulate import tabulate
from .models import ExecTimeStats

Logger = logging.getLogger("benchmark")


def print_summary(stats: list[ExecTimeStats]) -> None:
    Logger.info("-" * 100)
    print("\n* All values are provided in seconds")

    headers = ["Host", "Model", "Mean", "SD", "Median", "Min", "Max", "Sample size"]
    print(tabulate(stats, headers=headers, tablefmt="simple_outline"))  # type: ignore
