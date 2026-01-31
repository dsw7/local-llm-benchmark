from os import path
from tomllib import load, TOMLDecodeError
from .models import Configs
from .utils import BenchmarkError


def _clamp_num_rounds(rounds: int) -> int:
    # minimum of 2 rounds needed to calculate standard deviation
    return max(2, min(rounds, 10))


def check_and_load_config() -> Configs:
    config_file = "configs.toml"

    if not path.exists(config_file):
        raise BenchmarkError(f"The file {config_file} does not exist.")

    with open(config_file, "rb") as f:
        try:
            config_data = load(f)
        except TOMLDecodeError as e:
            raise BenchmarkError("Configurations can't be decoded", e) from e

    servers = [f'{s["host"]}:{s["port"]}' for s in config_data["servers"]]

    try:
        configs = Configs(
            prompt=config_data["misc"]["prompt"],
            model=config_data["misc"]["model"],
            rounds=_clamp_num_rounds(config_data["misc"]["rounds"]),
            servers=servers,
        )
    except KeyError as e:
        raise BenchmarkError("One or more configurations is missing", e) from e

    return configs
