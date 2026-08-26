from functools import cache
from logging import getLogger

from colorama import Back, Style
from requests import get, exceptions
from tabulate import tabulate
from ollama import Client

from .exceptions import BenchmarkError
from .models import Configs, ExecutionTimes

Logger = getLogger("benchmark")


@cache
def _get_client(host: str) -> Client:
    return Client(host)


def _check_server_up(server: str) -> None:
    get(f"http://{server}", timeout=5)


def _check_model_exists_(server: str, model: str) -> None:
    client = _get_client(server)
    response = client.list()

    for list_model in response.models:
        if list_model.model == model:
            break
    else:
        raise BenchmarkError(
            f"Model '{model}' not found on server '{server.split(':')[0]}'"
        )


def _preload_model(server: str, model: str) -> None:
    client = _get_client(server)
    Logger.info("Preloading %s on server %s", model, server)

    client.generate(model=model, prompt="What is 3 + 5?", keep_alive="30m")


def _run_and_time_query(host: str, prompt: str, model: str) -> float:
    client = _get_client(host)

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
        _check_server_up(configs.server)
    except exceptions.ConnectionError as e:
        raise BenchmarkError(str(e)) from e

    _check_model_exists_(configs.server, configs.model)
    _preload_model(configs.server, configs.model)

    try:
        exec_times: ExecutionTimes = _run_and_time_queries(configs)
    except KeyboardInterrupt as e:
        raise BenchmarkError("\nBenchmarking was manually aborted!") from e

    _print_summary_to_stdout(configs, exec_times)
