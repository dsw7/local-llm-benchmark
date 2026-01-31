from statistics import mean, stdev
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
    median: float
    min_val: float
    model: str
    sample_size: int

    def get_mean_exec_time(self, ndigits: int | None = None) -> float:
        mean_val = mean(self.exec_times)

        if ndigits is None:
            return mean_val

        return round(mean_val, ndigits)

    def get_stdev_exec_time(self, ndigits: int | None = None) -> float:
        stdev_val = stdev(self.exec_times)

        if ndigits is None:
            return stdev_val

        return round(stdev_val, ndigits)

    def get_plot_filename(self) -> str:
        return f"results_{self.host.replace(':', '_')}.pdf"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
