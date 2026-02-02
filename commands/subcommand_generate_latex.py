from subprocess import run, CalledProcessError

from core.consts import DIR_OUTPUT
from core.dataclass_json_io import load_stats_models_from_json
from core.exceptions import BenchmarkError
from core.models import Benchmark, ExecutionTimes

_DIR_LATEX_FILES = DIR_OUTPUT / "latex"


def _assemble_technical_details_section(benchmark_obj: Benchmark) -> str:
    return rf"""\section{{Technical details}}
\subsection*{{Basic test parameters:}}
\begin{{tabularx}}{{\textwidth}}{{XX}}
  Acquisition date & \today \\
  Acquisition time & \currenttime \\
  Model & {benchmark_obj.model} \\
  Sample size & {benchmark_obj.sample_size} \\
\end{{tabularx}}

\subsection*{{Prompt:}}
\begin{{verbatim}}
{benchmark_obj.prompt}
\end{{verbatim}}
\newpage
"""


def _assemble_host_section(entry: ExecutionTimes) -> str:
    return rf"""\section{{{entry.host}}}
\subsection*{{Statistics:}}
\begin{{tabularx}}{{\textwidth}}{{XX}}
  Mean execution time & {entry.get_mean_exec_time(ndigits=3)} s \\
  Stardard deviation of execution time & {entry.get_stdev_exec_time(ndigits=3)} s \\
  Median execution time & {entry.get_median_exec_time(ndigits=3)} s \\
  Minimum execution time & {entry.get_min_exec_time(ndigits=3)} s \\
  Maximum execution time & {entry.get_max_exec_time(ndigits=3)} s \\
\end{{tabularx}}
\newpage
"""


def _assemble_host_sections(benchmark_obj: Benchmark) -> str:
    text = ""

    for entry in benchmark_obj.exec_times_per_host:
        text += _assemble_host_section(entry)

    return text


def _assemble_full_text(benchmark_obj: Benchmark) -> str:
    return rf"""\documentclass[10pt]{{article}}

\usepackage{{courier}}
\usepackage{{datetime}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{tabularx}}
\usepackage{{titlesec}}

\geometry{{margin=1in}}
\titleformat{{\section}}{{\bfseries\large}}{{}}{{0em}}{{}}[\titlerule]
\renewcommand*\familydefault{{\ttdefault}}

\begin{{document}}
\tableofcontents
\newpage

{_assemble_technical_details_section(benchmark_obj)}
{_assemble_host_sections(benchmark_obj)}
\end{{document}}
"""


def main() -> None:
    if not _DIR_LATEX_FILES.exists():
        _DIR_LATEX_FILES.mkdir()

    try:
        benchmark_obj: Benchmark = load_stats_models_from_json()
    except BenchmarkError as e:
        raise SystemExit(e) from e

    path_report_tex = _DIR_LATEX_FILES / "report.tex"
    report = _assemble_full_text(benchmark_obj)
    path_report_tex.write_text(report)

    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        f"-output-directory={_DIR_LATEX_FILES}",
        f"{path_report_tex}",
    ]
    try:
        run(command, check=True)
    except CalledProcessError as e:
        raise SystemExit(e) from e


if __name__ == "__main__":
    main()
