from .exceptions import BenchmarkError, ConfigError
from .export_normal_distribution_plots import export_normal_distribution_plots
from .load_configs import check_and_load_config
from .run_benchmarks import run_benchmarks
from .consts import OutputDirectory, PlotsDirectory

__all__ = [
    "BenchmarkError",
    "ConfigError",
    "check_and_load_config",
    "run_benchmarks",
    "export_normal_distribution_plots",
]

if not OutputDirectory.exists():
    OutputDirectory.mkdir()

if not PlotsDirectory.exists():
    PlotsDirectory.mkdir()
