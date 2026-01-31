from requests import get, exceptions
from .models import Configs
from .utils import BenchmarkError


def _check_servers_up(servers: list[str]) -> None:
    for server in servers:
        get(f"http://{server}", timeout=5)


def run_benchmarks(configs: Configs) -> None:
    try:
        _check_servers_up(configs.servers)
    except exceptions.ConnectionError as e:
        raise BenchmarkError(str(e)) from e
