from dataclasses import dataclass


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
