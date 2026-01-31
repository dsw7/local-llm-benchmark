from logging import getLogger
from statistics import mean, stdev, median
from time import time
from colorama import Back, Style
from requests import get, exceptions
from .models import Configs, ExecTimeStats
from .utils import BenchmarkError, get_client

Logger = getLogger("benchmark")


def _check_servers_up(servers: list[str]) -> None:
    for server in servers:
        get(f"http://{server}", timeout=5)


def _check_models_exist(servers: list[str], model: str) -> None:
    for server in servers:
        client = get_client(server)
        response = client.list()

        for list_model in response.models:
            if list_model.model == model:
                break
        else:
            raise BenchmarkError(
                f"Model '{model}' not found on server '{server.split(':')[0]}'"
            )


def _preload_models(servers: list[str], model: str) -> None:
    for server in servers:
        client = get_client(server)
        Logger.info("Preloading %s on server %s", model, server)

        client.generate(model=model, prompt="What is 3 + 5?", keep_alive="30m")


def _run_and_time_query(host: str, prompt: str, model: str) -> float:
    client = get_client(host)

    time_start = time()
    stream = client.generate(model=model, prompt=prompt, stream=True)

    for chunk in stream:
        print(chunk["response"], end="", flush=True)

    return time() - time_start


def _run_and_time_queries(
    servers: list[str], num_rounds: int, prompt: str, model: str
) -> list[ExecTimeStats]:
    results = []

    for server in servers:
        exec_times = []

        for run in range(1, num_rounds + 1):
            Logger.info("-" * 100)
            Logger.info(
                Back.GREEN + f"Run {run} | {server} | {model}" + Style.RESET_ALL
            )
            exec_time = _run_and_time_query(server, prompt, model)
            Logger.info(f"Execution time: {exec_time:.3f}s")
            exec_times.append(exec_time)

        results.append(
            ExecTimeStats(
                exec_times=exec_times,
                host=server,
                max_val=max(exec_times),
                mean=round(mean(exec_times), 5),
                median=round(median(exec_times), 5),
                min_val=min(exec_times),
                model=model,
                sample_size=len(exec_times),
                stdev=round(stdev(exec_times), 5),
            )
        )

    return results


def run_benchmarks(configs: Configs) -> None:
    try:
        _check_servers_up(configs.servers)
    except exceptions.ConnectionError as e:
        raise BenchmarkError(str(e)) from e

    _check_models_exist(configs.servers, configs.model)
    _preload_models(configs.servers, configs.model)

    try:
        stats: list[ExecTimeStats] = _run_and_time_queries(
            configs.servers, configs.rounds, configs.prompt, configs.model
        )
    except KeyboardInterrupt as e:
        raise BenchmarkError("\nBenchmarking was manually aborted!") from e
