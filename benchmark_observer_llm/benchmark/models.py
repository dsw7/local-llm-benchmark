from dataclasses import dataclass
from statistics import mean, stdev, median


@dataclass
class Configs:
    model: str
    prompt: str
    rounds: int
    server: str


@dataclass
class ExecutionTimes:
    exec_times: list[float]
    host: str

    def get_pdf_name_from_host(self) -> str:
        return f"results_{self.host.replace(':', '_')}.pdf"

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

    def get_median_exec_time(self, ndigits: int | None = None) -> float:
        median_val = median(self.exec_times)

        if ndigits is None:
            return median_val

        return round(median_val, ndigits)

    def get_min_exec_time(self, ndigits: int | None = None) -> float:
        min_val = min(self.exec_times)

        if ndigits is None:
            return min_val

        return round(min_val, ndigits)

    def get_max_exec_time(self, ndigits: int | None = None) -> float:
        max_val = max(self.exec_times)

        if ndigits is None:
            return max_val

        return round(max_val, ndigits)


@dataclass
class Benchmark:
    exec_times_per_host: list[ExecutionTimes]
    model: str
    prompt: str
    sample_size: int
    timestamp: str
