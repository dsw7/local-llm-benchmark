from logging import getLogger
from tabulate import tabulate
from .models import ExecTimeStats

Logger = getLogger("benchmark")


def print_summary_to_stdout(stats: list[ExecTimeStats]) -> None:
    Logger.info("-" * 100)
    print("\n* All values are provided in seconds")

    # transpose data to match the headers list
    stats_transposed = [
        [
            s.host,
            s.model,
            s.get_mean_exec_time(ndigits=5),
            s.get_stdev_exec_time(ndigits=5),
            s.get_median_exec_time(ndigits=5),
            s.min_val,
            s.max_val,
            s.sample_size,
        ]
        for s in stats
    ]

    headers = ["Host", "Model", "Mean", "SD", "Median", "Min", "Max", "Sample size"]
    print(tabulate(stats_transposed, headers=headers, tablefmt="simple_outline"))
    Logger.info("-" * 100)
