from logging import getLogger
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.stats import norm
from numpy import linspace
from .models import ExecTimeStats

_PLOT_FONT_SIZE = 8
_PLOT_WIDTH = 5  # inches
_PLOT_HEIGHT = 3  # inches
_PLOT_DIRECTORY = Path("plots")

Logger = getLogger("benchmark")

plt.rcParams.update(
    {
        "font.size": _PLOT_FONT_SIZE,
        "font.family": "monospace",
        "font.monospace": ["Courier", "Courier New", "DejaVu Sans Mono"],
    }
)


def _get_plot_filename(host_str: str) -> Path:
    host_sub = host_str.replace(":", "_")
    return _PLOT_DIRECTORY / f"results_{host_sub}.pdf"


def _plot_normal_distribution(stats: ExecTimeStats) -> Path:
    mu = stats.mean
    sigma = stats.stdev

    x = linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    f_x = norm.pdf(x, mu, sigma)
    f_exec_times = norm.pdf(stats.exec_times, mu, sigma)

    plt.figure(figsize=(_PLOT_WIDTH, _PLOT_HEIGHT))
    plt.plot(x, f_x, alpha=0.5, c="k", lw=0.5)
    plt.scatter(stats.exec_times, f_exec_times.tolist(), c="k", s=12, marker="x")
    plt.xlabel("Time (s)")

    ax = plt.gca()
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path_to_plot = _get_plot_filename(stats.host)
    Logger.info("Exporting plot for host %s to %s", stats.host, path_to_plot)

    plt.tight_layout()
    plt.savefig(path_to_plot)
    return path_to_plot


def generate_report(list_stats: list[ExecTimeStats]) -> None:
    if not _PLOT_DIRECTORY.exists():
        Logger.info("Creating new directory: %s", _PLOT_DIRECTORY)
        _PLOT_DIRECTORY.mkdir()

    plots: list[Path] = []

    for stats in list_stats:
        plots.append(_plot_normal_distribution(stats))
