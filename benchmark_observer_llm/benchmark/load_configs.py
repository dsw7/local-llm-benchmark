from os import path
from tomllib import load, TOMLDecodeError

from .exceptions import ConfigError
from .models import Configs


def _clamp_num_rounds(rounds: int) -> int:
    # minimum of 2 rounds needed to calculate standard deviation
    return max(2, min(rounds, 250))


def check_and_load_config() -> Configs:
    config_file = "configs.toml"

    if not path.exists(config_file):
        raise ConfigError(f"The file {config_file} does not exist.")

    with open(config_file, "rb") as f:
        try:
            config_data = load(f)
        except TOMLDecodeError as e:
            raise ConfigError(f"Configurations can't be decoded: {e}") from e

    server = f'{config_data["server"]["host"]}:{config_data["server"]["port"]}'

    try:
        configs = Configs(
            model=config_data["misc"]["model"],
            rounds=_clamp_num_rounds(config_data["misc"]["rounds"]),
            server=server,
        )
    except KeyError as e:
        raise ConfigError("One or more configurations is missing", e) from e

    return configs
