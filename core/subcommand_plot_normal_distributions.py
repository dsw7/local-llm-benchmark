from logging import getLogger
from pathlib import Path

from matplotlib import pyplot as plt
from numpy import linspace
from scipy.stats import norm

from .consts import PlotsDirectory
from .dataclass_json_io import load_stats_models_from_json
from .models import ExecTimeStats

_PLOT_FONT_SIZE = 8
_PLOT_WIDTH = 5  # inches
_PLOT_HEIGHT = 3  # inches

Logger = getLogger("benchmark")

plt.rcParams.update(
    {
        "font.size": _PLOT_FONT_SIZE,
        "font.family": "monospace",
        "font.monospace": ["Courier", "Courier New", "DejaVu Sans Mono"],
    }
)


def _get_plot_filename(host: str) -> Path:
    return PlotsDirectory / f"results_{host.replace(':', '_')}.pdf"


def _plot_normal_distribution(stats: ExecTimeStats) -> None:
    mu = stats.get_mean_exec_time()
    sigma = stats.get_stdev_exec_time()

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


def _export_normal_distribution_plots() -> None:
    stats, _ = load_stats_models_from_json()

    for s in stats:
        _plot_normal_distribution(s)

def main() -> None:
    try:
        _export_normal_distribution_plots()
    except BenchmarkError as e:
        raise SystemExit from e


if __name__ == "__main__":
    main()
