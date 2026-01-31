from dataclasses import dataclass, asdict
from typing import Any


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

    def get_plot_filename(self) -> str:
        return f"results_{self.host.replace(':', '_')}.pdf"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
