from logging import getLogger

from matplotlib import pyplot as plt
from numpy import linspace
from scipy.stats import norm

from core.consts import DIR_PLOTS
from core.dataclass_json_io import load_stats_models_from_json
from core.exceptions import BenchmarkError
from core.models import Benchmark, ExecutionTimes

_PLOT_FONT_SIZE = 8
_PLOT_WIDTH = 5  # inches
_PLOT_HEIGHT = 3  # inches

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.left": False,
        "figure.figsize": [_PLOT_WIDTH, _PLOT_HEIGHT],
        "font.family": "monospace",
        "font.monospace": ["Courier", "Courier New", "DejaVu Sans Mono"],
        "font.size": _PLOT_FONT_SIZE,
        "ytick.labelleft": False,
        "ytick.left": False,
    }
)


def _plot_theoretical_normal_curve(mu: float, sigma: float) -> None:
    x = linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    f_x = norm.pdf(x, mu, sigma)
    plt.plot(x, f_x, alpha=0.5, c="k", lw=0.5)


def _plot_histogram(exec_times: list[float]) -> None:
    plt.hist(
        exec_times,
        bins=30,
        density=True,
        color="g",
        alpha=0.5,
        edgecolor="white",
        lw=0.25,
    )


def _plot_normal_distribution(entry: ExecutionTimes) -> None:
    mu = entry.get_mean_exec_time()
    sigma = entry.get_stdev_exec_time()

    plt.figure()
    _plot_theoretical_normal_curve(mu, sigma)
    _plot_histogram(entry.exec_times)

    plt.xlabel("Time (s)")
    plt.ylabel("Density")

    path_to_plot = DIR_PLOTS / entry.get_pdf_name_from_host()
    getLogger("benchmark").info(
        "Exporting plot for host %s to %s", entry.host, path_to_plot
    )

    plt.tight_layout()
    plt.savefig(path_to_plot)


def _export_normal_distribution_plots() -> None:
    benchmark_obj: Benchmark = load_stats_models_from_json()

    for exec_times in benchmark_obj.exec_times_per_host:
        _plot_normal_distribution(exec_times)


def main() -> None:
    if not DIR_PLOTS.exists():
        DIR_PLOTS.mkdir()

    try:
        _export_normal_distribution_plots()
    except BenchmarkError as e:
        raise SystemExit(e) from e


if __name__ == "__main__":
    main()
