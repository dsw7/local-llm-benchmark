from pathlib import Path
from dataclasses import asdict
from json import dumps
from .models import ExecTimeStats

OutputFile = Path("stats.json")


def dump_stats_models_to_json(stats: list[ExecTimeStats], prompt: str) -> None:
    servers = []

    for s in stats:
        servers.append({s.host: asdict(s)})

    json = {
        "prompt": prompt,
        "servers": servers,
    }

    OutputFile.write_text(dumps(json, indent=4))
