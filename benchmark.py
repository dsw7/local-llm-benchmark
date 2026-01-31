#!/usr/bin/env python3

import logging
import sys
from statistics import mean, stdev, median
from time import time
from colorama import Back, Style
from tabulate import tabulate
import requests
import core
from core.models import ExecTimeStats
from core.reporting import generate_report
from core.utils import get_client

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
Logger = logging.getLogger("benchmark")


def check_servers_up(servers: list[str]) -> None:
    for server in servers:
        requests.get(f"http://{server}", timeout=5)


def check_models_exist(servers: list[str], model: str) -> None:
    for server in servers:
        client = get_client(server)
        response = client.list()

        for list_model in response.models:
            if list_model.model == model:
                break
        else:
            raise ValueError(
                f"Model '{model}' not found on server '{server.split(':')[0]}'"
            )


def preload_models(servers: list[str], model: str) -> None:
    for server in servers:
        client = get_client(server)
        Logger.info("Preloading %s on server %s", model, server)

        client.generate(model=model, prompt="What is 3 + 5?", keep_alive="30m")


def run_and_time_query(host: str, prompt: str, model: str) -> float:
    client = get_client(host)

    time_start = time()
    stream = client.generate(model=model, prompt=prompt, stream=True)

    for chunk in stream:
        print(chunk["response"], end="", flush=True)

    return time() - time_start


def run_and_time_queries(
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
            exec_time = run_and_time_query(server, prompt, model)
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


def print_summary(stats: list[ExecTimeStats]) -> None:
    Logger.info("-" * 100)
    print("\n* All values are provided in seconds")

    headers = ["Host", "Model", "Mean", "SD", "Median", "Min", "Max", "Sample size"]
    print(tabulate(stats, headers=headers, tablefmt="simple_outline"))  # type: ignore


def main() -> None:
    try:
        configs = core.check_and_load_config()
    except core.ConfigError as e:
        sys.exit(str(e))

    try:
        check_servers_up(configs.servers)
    except requests.exceptions.ConnectionError as e:
        sys.exit(str(e))

    try:
        check_models_exist(configs.servers, configs.model)
    except ValueError as e:
        sys.exit(str(e))

    preload_models(configs.servers, configs.model)

    try:
        stats: list[ExecTimeStats] = run_and_time_queries(
            configs.servers, configs.rounds, configs.prompt, configs.model
        )
    except KeyboardInterrupt:
        sys.exit("\nBenchmarking was manually aborted!")

    print_summary(stats)
    generate_report(stats)


if __name__ == "__main__":
    main()
