from dataclasses import dataclass
from pathlib import Path


@dataclass
class Configs:
    prompt: str
    model: str
    rounds: int
    servers: list[str]


@dataclass
class ExecTimeStats:
    exec_times: list[float]
    host: str
    max_val: float
    mean: float
    median: float
    min_val: float
    model: str
    sample_size: int
    stdev: float

    path_plot: Path | None = None
