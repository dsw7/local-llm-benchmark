from .exceptions import BenchmarkError, ConfigError
from .load_configs import check_and_load_config
from .run_benchmarks import run_benchmarks


def main() -> None:
    try:
        configs = check_and_load_config()
    except ConfigError as e:
        raise SystemExit(e) from e

    try:
        run_benchmarks(configs)
    except BenchmarkError as e:
        raise SystemExit(e) from e


if __name__ == "__main__":
    main()
