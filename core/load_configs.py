from dataclasses import dataclass
from os import path
from typing import Any
import tomllib


def _clamp_num_rounds(rounds: int) -> int:
    # minimum of 2 rounds needed to calculate standard deviation
    return max(2, min(rounds, 10))


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
            rounds=_clamp_num_rounds(config_data["misc"]["rounds"]),
            servers=servers,
        )
    except KeyError as e:
        raise ConfigError("One or more configurations is missing", e) from e

    return configs
