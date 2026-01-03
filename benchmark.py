#!/usr/bin/env python3

import functools
import logging
import sys
from dataclasses import dataclass
from statistics import mean, stdev, median
from time import time
from colorama import Back, Style
from ollama import Client
from tabulate import tabulate
import requests
import core

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ExecTimes:
    exec_times: list[float]
    host: str
    model: str


@dataclass
class ExecTimeStats:
    host: str
    model: str
    mean: float
    stdev: float
    median: float
    min_val: float
    max_val: float
    sample_size: int


@functools.cache
def get_client(host: str) -> Client:
    return Client(host)


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
        logger.info("Preloading %s on server %s", model, server)

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
) -> list[ExecTimes]:
    results = []

    for server in servers:
        exec_times = []

        for run in range(1, num_rounds + 1):
            logger.info("-" * 100)
            logger.info(
                Back.GREEN + f"Run {run} | {server} | {model}" + Style.RESET_ALL
            )
            exec_time = run_and_time_query(server, prompt, model)
            logger.info(f"Execution time: {exec_time:.3f}s")
            exec_times.append(exec_time)

        results.append(ExecTimes(exec_times=exec_times, host=server, model=model))

    return results


def get_stats_from_exec_times(results: list[ExecTimes]) -> list[ExecTimeStats]:
    stats = []

    for item in results:
        mean_val = round(mean(item.exec_times), 5)
        stdev_val = round(stdev(item.exec_times), 5)
        median_val = round(median(item.exec_times), 5)

        stats.append(
            ExecTimeStats(
                host=item.host,
                max_val=max(item.exec_times),
                mean=mean_val,
                median=median_val,
                min_val=min(item.exec_times),
                model=item.model,
                sample_size=len(item.exec_times),
                stdev=stdev_val,
            )
        )

    return stats


def print_summary(stats: list[ExecTimeStats]) -> None:
    logger.info("-" * 100)
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
        exec_times: list[ExecTimes] = run_and_time_queries(
            configs.servers, configs.rounds, configs.prompt, configs.model
        )
    except KeyboardInterrupt:
        sys.exit("\nBenchmarking was manually aborted!")

    stats: list[ExecTimeStats] = get_stats_from_exec_times(exec_times)
    print_summary(stats)


if __name__ == "__main__":
    main()
