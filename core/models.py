from statistics import mean, stdev, median
from dataclasses import dataclass, asdict
from json import dump


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
    model: str
    prompt: str
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

    def get_median_exec_time(self, ndigits: int | None = None) -> float:
        median_val = median(self.exec_times)

        if ndigits is None:
            return median_val

        return round(median_val, ndigits)

    def get_min_exec_time(self) -> float:
        return min(self.exec_times)

    def get_max_exec_time(self) -> float:
        return max(self.exec_times)

    def get_plot_filename(self) -> str:
        return f"results_{self.host.replace(':', '_')}.pdf"

    def get_json_filename(self) -> str:
        return f"results_{self.host.replace(':', '_')}.json"


def dump_stats_to_json(stats: list[ExecTimeStats]) -> list[str]:
    paths = []

    for s in stats:
        file_path = s.get_json_filename()
        paths.append(file_path)

        with open(file_path, "w") as file:
            dump(asdict(s), file, indent=4)

    return paths
