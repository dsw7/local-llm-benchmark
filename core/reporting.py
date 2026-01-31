import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import linspace
from .models import ExecTimeStats

_PLOT_FONT_SIZE = 8
_PLOT_WIDTH = 5  # inches
_PLOT_HEIGHT = 3  # inches

plt.rcParams.update(
    {
        "font.size": _PLOT_FONT_SIZE,
        "font.family": "monospace",
        "font.monospace": ["Courier", "Courier New", "DejaVu Sans Mono"],
    }
)


def _plot_normal_distribution(stats: ExecTimeStats) -> str:
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
    filename = f"results_{stats.host}.png"
    plt.savefig(filename)
    return filename


def generate_report(stats: list[ExecTimeStats]) -> None:
    for s in stats:
        _plot_normal_distribution(s)
