from dataclasses import asdict
from json import dumps, loads

from .consts import DIR_OUTPUT
from .exceptions import BenchmarkError
from .models import ExecTimeStats

OutputFile = DIR_OUTPUT / "stats.json"


def dump_stats_models_to_json(stats: list[ExecTimeStats], prompt: str) -> None:
    servers = []

    for s in stats:
        servers.append(asdict(s))

    json = {
        "prompt": prompt,
        "servers": servers,
    }

    OutputFile.write_text(dumps(json, indent=4))


def load_stats_models_from_json() -> tuple[list[ExecTimeStats], str]:
    if not OutputFile.exists():
        raise BenchmarkError(f"{OutputFile} does not exist. Cannot proceed")

    contents = loads(OutputFile.read_text())

    stats = []

    for s in contents["servers"]:
        stats.append(
            ExecTimeStats(
                exec_times=s["exec_times"],
                host=s["host"],
                model=s["model"],
                sample_size=s["exec_times"],
            )
        )

    return stats, contents["prompt"]
