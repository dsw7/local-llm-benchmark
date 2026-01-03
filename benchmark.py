#!/usr/bin/env python3

import functools
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev
from time import time
from ollama import Client
import requests
import core

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(asctime)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Stats:
    exec_time: float
    host: str
    model: str


@dataclass
class Summary:
    host: str
    mean: float
    model: str
    stdev: float


@functools.cache
def get_client(host: str) -> Client:
    return Client(host)


def check_servers_up(servers: list[str]) -> None:
    for server in servers:
        requests.get(f"http://{server}", timeout=5)


def check_model_exists(servers: list[str], model: str) -> None:
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


def run_queries(host: str, prompt: str, model: str) -> Stats:
    client = get_client(host)

    time_start = time()
    stream = client.generate(model=model, prompt=prompt, stream=True)

    logger.info("-" * 100)
    logger.info(f"[{host}] [{model}]")

    for chunk in stream:
        print(chunk["response"], end="", flush=True)

    total_time = round(time() - time_start, 2)
    logger.info(f"Execution time: {total_time}s")

    return Stats(exec_time=total_time, host=host, model=model)


def process_stats(stats: list[Stats]) -> list[Summary]:
    grouped_data = defaultdict(list)
    for stat in stats:
        key = (stat.host, stat.model)
        grouped_data[key].append(stat.exec_time)

    summary = []

    for key, exec_times in grouped_data.items():
        filtered_times = core.reject_outliers(exec_times)

        if len(filtered_times) < 2:
            mean_val = mean(filtered_times)
            stdev_val = 0.0
        else:
            mean_val = mean(filtered_times)
            stdev_val = stdev(filtered_times)

        summary.append(
            Summary(
                host=key[0],
                mean=round(mean_val, 3),
                model=key[1],
                stdev=round(stdev_val, 3),
            )
        )

    return summary


def print_summary(summary: list[Summary]) -> None:
    logger.info("-" * 100)
    print(f"{'Host':<20}{'Model':<30}{'Mean (s)':<20}{'SD (s)':<20}")

    for item in summary:
        print(f"{item.host:<20}{item.model:<30}{item.mean:<20}{item.stdev:<20}")


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
        check_model_exists(configs.servers, configs.model)
    except ValueError as e:
        sys.exit(str(e))

    stats: list[Stats] = []

    try:
        for server in configs.servers:
            for _ in range(configs.rounds):
                stats.append(run_queries(server, configs.prompt, configs.model))
    except KeyboardInterrupt:
        sys.exit("\nBenchmarking was manually aborted!")

    summary: list[Summary] = process_stats(stats)
    print_summary(summary)


if __name__ == "__main__":
    main()
