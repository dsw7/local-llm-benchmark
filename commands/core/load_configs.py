from os import path
from pathlib import Path
from tomllib import load, TOMLDecodeError

from .exceptions import ConfigError
from .models import Configs


def _clamp_num_rounds(rounds: int) -> int:
    # minimum of 2 rounds needed to calculate standard deviation
    return max(2, min(rounds, 50))


def check_and_load_config() -> Configs:
    config_file = "configs.toml"

    if not path.exists(config_file):
        raise ConfigError(f"The file {config_file} does not exist.")

    with open(config_file, "rb") as f:
        try:
            config_data = load(f)
        except TOMLDecodeError as e:
            raise ConfigError(f"Configurations can't be decoded: {e}") from e

    servers = [f'{s["host"]}:{s["port"]}' for s in config_data["servers"]]

    report_dump_location: Path | None = None

    if "report_dump_location" in config_data["misc"]:
        report_dump_location = Path(config_data["misc"]["report_dump_location"])

    try:
        configs = Configs(
            model=config_data["misc"]["model"],
            prompt=config_data["misc"]["prompt"],
            report_dump_location=report_dump_location,
            rounds=_clamp_num_rounds(config_data["misc"]["rounds"]),
            servers=servers,
        )
    except KeyError as e:
        raise ConfigError("One or more configurations is missing", e) from e

    return configs
