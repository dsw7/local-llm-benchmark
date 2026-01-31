import pathlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import linspace
from .models import ExecTimeStats

_PLOT_FONT_SIZE = 8
_PLOT_WIDTH = 5  # inches
_PLOT_HEIGHT = 3  # inches
_PLOT_DIRECTORY = pathlib.Path("plots")

plt.rcParams.update(
    {
        "font.size": _PLOT_FONT_SIZE,
        "font.family": "monospace",
        "font.monospace": ["Courier", "Courier New", "DejaVu Sans Mono"],
    }
)


def _plot_normal_distribution(stats: ExecTimeStats) -> pathlib.Path:
    data = []
    mu = stats.mean
    sigma = stats.stdev

    x = linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    y = norm.pdf(x, mu, sigma)

    y2 = norm.pdf(data, mu, sigma)

    plt.figure(figsize=(_PLOT_WIDTH, _PLOT_HEIGHT))
    plt.plot(x, y, alpha=0.5, c="k", lw=0.5)
    plt.scatter(data, y2.tolist(), c="k", s=12, marker="x")
    plt.xlabel("Time (s)")

    ax = plt.gca()
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    path_to_plot = _PLOT_DIRECTORY / f"results_{stats.host}.png"
    plt.savefig(path_to_plot)

    return path_to_plot


def generate_report(list_stats: list[ExecTimeStats]) -> None:
    if not _PLOT_DIRECTORY.exists():
        _PLOT_DIRECTORY.mkdir()

    plots: list[pathlib.Path] = []

    for stats in list_stats:
        plots.append(_plot_normal_distribution(stats))
