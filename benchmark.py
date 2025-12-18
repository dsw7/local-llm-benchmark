#!/usr/bin/env python3

import functools
import logging
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from os import path
from statistics import mean, stdev
from time import time
from typing import Any
from ollama import Client
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(asctime)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@functools.cache
def get_client(host: str) -> Client:
    return Client(host)


def clamp_num_rounds(rounds: int) -> int:
    # minimum of 2 rounds needed to calculate standard deviation
    return max(2, min(rounds, 10))


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


@dataclass
class Configs:
    prompt: str
    model: str
    rounds: int
    servers: list[str]


class ConfigError(Exception):
    def __init__(self, message: str, *args: Any):
        self.message = message
        self.args = args

    def __str__(self) -> str:
        return f"ConfigError: {self.message}"


def check_and_load_config() -> Configs:
    config_file = "configs.toml"

    if not path.exists(config_file):
        raise ConfigError(f"The file {config_file} does not exist.")

    with open(config_file, "rb") as f:
        try:
            config_data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError("Configurations can't be decoded", e) from e

    servers = [f'{s["host"]}:{s["port"]}' for s in config_data["servers"]]

    try:
        configs = Configs(
            prompt=config_data["misc"]["prompt"],
            model=config_data["misc"]["model"],
            rounds=clamp_num_rounds(config_data["misc"]["rounds"]),
            servers=servers,
        )
    except KeyError as e:
        raise ConfigError("One or more configurations is missing", e) from e

    return configs


@dataclass
class Stats:
    exec_time: float
    host: str
    model: str


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


def reject_outliers(data: list[float], m: int = 2) -> list[float]:
    mean_val = mean(data)
    stdev_val = stdev(data)
    return [x for x in data if abs(x - mean_val) < m * stdev_val]


def process_stats(stats: list[Stats]) -> None:
    grouped_data = defaultdict(list)
    for stat in stats:
        key = (stat.host, stat.model)
        grouped_data[key].append(stat.exec_time)

    results = {}

    for key, exec_times in grouped_data.items():
        filtered_times = reject_outliers(exec_times)

        if len(filtered_times) < 2:
            mean_val = mean(filtered_times)
            stdev_val = 0.0
        else:
            mean_val = mean(filtered_times)
            stdev_val = stdev(filtered_times)

        results[key] = {"mean": mean_val, "stdev": stdev_val}

    logger.info("-" * 100)
    print(f"{'Host':<20}{'Model':<30}{'Mean (s)':<20}{'SD (s)':<20}")

    for key, value in results.items():
        host, model = key
        _mean, _stdev = round(value["mean"], 3), round(value["stdev"], 3)
        print(f"{host:<20}{model:<30}{_mean:<20}{_stdev:<20}")


def main() -> None:
    try:
        configs = check_and_load_config()
    except ConfigError as e:
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

    process_stats(stats)


if __name__ == "__main__":
    main()
