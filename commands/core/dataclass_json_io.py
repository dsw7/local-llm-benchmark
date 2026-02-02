from dataclasses import asdict
from json import dumps, loads

from .consts import DIR_OUTPUT
from .exceptions import BenchmarkError
from .models import Benchmark, ExecutionTimes

_BENCHMARK_JSON = DIR_OUTPUT / "benchmark.json"


def dump_stats_models_to_json(benchmark_obj: Benchmark) -> None:
    json = {
        "exec_times_per_host": [asdict(s) for s in benchmark_obj.exec_times_per_host],
        "model": benchmark_obj.model,
        "prompt": benchmark_obj.prompt,
        "sample_size": benchmark_obj.sample_size,
    }

    _BENCHMARK_JSON.write_text(dumps(json, indent=4))


def load_stats_models_from_json() -> Benchmark:
    if not _BENCHMARK_JSON.exists():
        raise BenchmarkError(f"{_BENCHMARK_JSON} does not exist. Cannot proceed")

    json = loads(_BENCHMARK_JSON.read_text())

    return Benchmark(
        exec_times_per_host=[
            ExecutionTimes(
                exec_times=s["exec_times"],
                host=s["host"],
            )
            for s in json["exec_times_per_host"]
        ],
        model=json["model"],
        prompt=json["prompt"],
        sample_size=json["sample_size"],
    )
