from logging import getLogger
from typing import Literal

from matplotlib import pyplot as plt
from numpy import linspace
from scipy.stats import norm

from core.consts import DIR_PLOTS, PATH_BOXPLOT
from core.dataclass_json_io import load_stats_models_from_json
from core.exceptions import BenchmarkError
from core.models import Benchmark, ExecutionTimes

_PLOT_FONT_SIZE = 8
_PLOT_WIDTH = 5  # inches
_PLOT_HEIGHT = 3  # inches

Logger = getLogger("benchmark")

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.figsize": [_PLOT_WIDTH, _PLOT_HEIGHT],
        "font.family": "monospace",
        "font.monospace": ["Courier", "Courier New", "DejaVu Sans Mono"],
        "font.size": _PLOT_FONT_SIZE,
    }
)


def _plot_theoretical_normal_curve(mu: float, sigma: float) -> None:
    x = linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    f_x = norm.pdf(x, mu, sigma)
    plt.plot(x, f_x, alpha=0.5, c="k", lw=0.5)


def _plot_histogram(exec_times: list[float]) -> None:
    hist_rv = plt.hist(
        exec_times,
        alpha=0.5,
        bins=30,
        color="g",
        density=True,
        edgecolor="white",
        lw=0.25,
    )
    bins = hist_rv[1]
    Logger.info("The histogram bin width is %f seconds", bins[1] - bins[0])


def _plot_normal_distribution(entry: ExecutionTimes) -> None:
    Logger.info("Plotting normal distribution for host %s", entry.host)
    mu = entry.get_mean_exec_time()
    sigma = entry.get_stdev_exec_time()

    plt.figure()
    _plot_theoretical_normal_curve(mu, sigma)
    _plot_histogram(entry.exec_times)

    plt.xlabel("Inference time (s)")
    plt.ylabel("Density")

    path_to_plot = DIR_PLOTS / entry.get_pdf_name_from_host()
    Logger.info(
        "Exporting normal distribution for host %s to %s", entry.host, path_to_plot
    )

    plt.tight_layout()
    plt.savefig(path_to_plot)


def _plot_boxplots(entries: list[ExecutionTimes]) -> None:
    exec_times = []
    hosts = []

    for entry in entries:
        exec_times.append(entry.exec_times)
        hosts.append(entry.host)

    plt.figure()

    orientation: Literal["vertical", "horizontal"]

    if len(entries) < 3:
        plt.ylabel("Inference time (s)")
        orientation = "vertical"
    else:
        plt.xlabel("Inference time (s)")
        orientation = "horizontal"

    plt.boxplot(exec_times, tick_labels=hosts, orientation=orientation)

    Logger.info("Exporting boxplot to file %s", PATH_BOXPLOT)
    plt.tight_layout()
    plt.savefig(PATH_BOXPLOT)


def _export_all_plots() -> None:
    benchmark_obj: Benchmark = load_stats_models_from_json()

    for exec_times in benchmark_obj.exec_times_per_host:
        _plot_normal_distribution(exec_times)

    _plot_boxplots(benchmark_obj.exec_times_per_host)


def main() -> None:
    if not DIR_PLOTS.exists():
        DIR_PLOTS.mkdir()

    try:
        _export_all_plots()
    except BenchmarkError as e:
        raise SystemExit(e) from e


if __name__ == "__main__":
    main()
