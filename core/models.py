from dataclasses import dataclass


@dataclass
class ExecTimes:
    exec_times: list[float]
    host: str
    model: str


@dataclass
class ExecTimeStats:
    host: str
    model: str
    mean: float
    stdev: float
    median: float
    min_val: float
    max_val: float
    sample_size: int
