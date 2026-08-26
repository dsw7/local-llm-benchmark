from logging import getLogger

from colorama import Back, Style
from requests import exceptions
from tabulate import tabulate

from .exceptions import BenchmarkError
from .models import Configs, ExecutionTimes
from .helpers import get_client, check_server_up, check_model_exists, preload_model

Logger = getLogger("benchmark")


def _run_and_time_query(host: str, prompt: str, model: str) -> float:
    client = get_client(host)

    response = client.generate(model=model, prompt=prompt, stream=False)

    if response.total_duration is None:
        raise BenchmarkError("Total duration is missing from response")

    return response.total_duration / 10**9


def _run_and_time_queries(configs: Configs) -> ExecutionTimes:
    Logger.info("Number of rounds per machine: %i", configs.rounds)

    exec_times = []

    for run in range(1, configs.rounds + 1):
        Logger.info(Back.GREEN + f" Run {run} " + Style.RESET_ALL)

        exec_time = _run_and_time_query(configs.server, configs.prompt, configs.model)
        Logger.info("Inference time: %.3fs", exec_time)

        exec_times.append(exec_time)

    return ExecutionTimes(exec_times=exec_times)


def _print_summary_to_stdout(configs: Configs, exec_times: ExecutionTimes) -> None:
    stats_transposed = [
        [
            configs.server,
            configs.model,
            exec_times.get_mean_exec_time(ndigits=5),
            exec_times.get_stdev_exec_time(ndigits=5),
            exec_times.get_median_exec_time(ndigits=5),
            exec_times.get_min_exec_time(),
            exec_times.get_max_exec_time(),
            configs.rounds,
        ]
    ]

    Logger.info("-" * 100)
    print("\n* All values are provided in seconds")

    headers = ["Host", "Model", "Mean", "SD", "Median", "Min", "Max", "Sample size"]
    print(tabulate(stats_transposed, headers=headers, tablefmt="simple_outline"))

    Logger.info("-" * 100)


def run_benchmarks(configs: Configs) -> None:
    try:
        check_server_up(configs.server)
    except exceptions.ConnectionError as e:
        raise BenchmarkError(str(e)) from e

    check_model_exists(configs.server, configs.model)

    Logger.info("Preloading model %s", configs.model)
    preload_model(configs.server, configs.model)

    try:
        exec_times: ExecutionTimes = _run_and_time_queries(configs)
    except KeyboardInterrupt as e:
        raise BenchmarkError("\nBenchmarking was manually aborted!") from e

    _print_summary_to_stdout(configs, exec_times)
