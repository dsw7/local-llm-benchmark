#!/usr/bin/env python3

import sys
import tomllib
from dataclasses import dataclass
from os import path
from time import time
from typing import Any
from ollama import Client


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


def clamp_num_rounds(rounds: int) -> int:
    return max(1, min(rounds, 10))


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


def run_queries(client: Client, prompt: str, model: str) -> None:
    time_start = time()
    stream = client.generate(model=model, prompt=prompt, stream=True)

    print("-" * 100)
    for chunk in stream:
        print(chunk["response"], end="", flush=True)

    total_time = round(time() - time_start, 2)
    print(f"\nExecution time: {total_time}s")


def main() -> None:
    try:
        configs = check_and_load_config()
    except ConfigError as e:
        sys.exit(str(e))

    clients = [Client(host=server) for server in configs.servers]

    for client in clients:
        for _ in range(configs.rounds):
            run_queries(client, configs.prompt, configs.model)


if __name__ == "__main__":
    main()
