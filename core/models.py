from dataclasses import dataclass, asdict
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

    path_to_plot: str | None = None

    def set_path_to_plot(self, path_to_plot: Path) -> None:
        self.path_to_plot = str(path_to_plot)

    def to_dict(self) -> dict:
        return asdict(self)
