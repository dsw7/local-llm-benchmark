from logging import getLogger
from pathlib import Path
from subprocess import run, CalledProcessError, PIPE

from core.consts import DIR_OUTPUT, DIR_PLOTS
from core.dataclass_json_io import load_stats_models_from_json
from core.exceptions import BenchmarkError
from core.models import Benchmark, ExecutionTimes

Logger = getLogger("benchmark")

_DIR_LATEX_FILES = DIR_OUTPUT / "latex"
_ENABLE_PDFLATEX_STDOUT = False


def _assemble_test_params_subsection(benchmark_obj: Benchmark) -> str:
    Logger.info("Assembling test parameters subsection")

    return rf"""\subsection*{{Test parameters}}
\begin{{tabularx}}{{\textwidth}}{{XX}}
  Acquisition date & \today \\
  Acquisition time & \currenttime \\
  Model & {benchmark_obj.model} \\
  Sample size & {benchmark_obj.sample_size} \\
\end{{tabularx}}
"""


def _assemble_prompt_subsection(prompt: str) -> str:
    Logger.info("Assembling prompt subsection")

    return rf"""\subsection*{{Prompt}}
\begin{{Verbatim}}[numbers=left, numbersep=2mm, frame=leftline, framesep=2mm]
{prompt.strip()}
\end{{Verbatim}}
"""


def _assemble_technical_details_section(benchmark_obj: Benchmark) -> str:
    Logger.info("Assembling technical details section")

    return rf"""\section{{Technical details}}
{_assemble_test_params_subsection(benchmark_obj)}
{_assemble_prompt_subsection(benchmark_obj.prompt)}
\newpage
"""


def _assemble_stats_subsection(entry: ExecutionTimes) -> str:
    Logger.info("Assembling statistics subsection for host %s", entry.host)

    return rf"""\subsection*{{Statistics}}
\begin{{tabularx}}{{\textwidth}}{{XX}}
  Mean execution time & {entry.get_mean_exec_time(ndigits=3)} s \\
  Stardard deviation of execution time & {entry.get_stdev_exec_time(ndigits=3)} s \\
  Median execution time & {entry.get_median_exec_time(ndigits=3)} s \\
  Minimum execution time & {entry.get_min_exec_time(ndigits=3)} s \\
  Maximum execution time & {entry.get_max_exec_time(ndigits=3)} s \\
\end{{tabularx}}
"""


def _assemble_normal_distribution_subsection(entry: ExecutionTimes) -> str:
    Logger.info("Assembling normal distribution subsection for host %s", entry.host)
    path_norm_dist = DIR_PLOTS / entry.get_pdf_name_from_host()

    if not path_norm_dist.exists():
        Logger.warning(
            "Cannot locate %s. Cannot add normal distribution", path_norm_dist
        )
        return r"""\subsection*{{Normal distribution}}
No data.
"""

    return rf"""\subsection*{{Normal distribution}}
\begin{{figure}}[ht]
  \centering
  \includegraphics{{{path_norm_dist}}}
  \caption{{Normal distribution for host {entry.host}}}
\end{{figure}}
"""


def _assemble_host_section(entry: ExecutionTimes) -> str:
    Logger.info("Assembling results section for host %s", entry.host)

    return rf"""\section{{{entry.host}}}
{_assemble_stats_subsection(entry)}
{_assemble_normal_distribution_subsection(entry)}
\newpage
"""


def _assemble_host_sections(benchmark_obj: Benchmark) -> str:
    sections = []

    for entry in benchmark_obj.exec_times_per_host:
        sections.append(_assemble_host_section(entry))

    return "\n".join(sections)


def _assemble_full_text(benchmark_obj: Benchmark) -> str:
    Logger.info("Assembling preamble and body")

    return rf"""\documentclass[10pt]{{article}}

\usepackage{{courier}}
\usepackage{{datetime}}
\usepackage{{fancyvrb}}
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


def _compile_latex_source(path_to_source: Path) -> None:
    Logger.info("Compiling LaTeX source %s", path_to_source)

    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        f"-output-directory={_DIR_LATEX_FILES}",
        f"{path_to_source}",
    ]

    try:
        process = run(command, stdout=PIPE, stderr=PIPE, check=True, text=True)
    except CalledProcessError as e:
        for line in e.stdout.splitlines():
            Logger.error(line)

        for line in e.stderr.splitlines():
            Logger.error(line)

        raise BenchmarkError(str(e)) from e

    if _ENABLE_PDFLATEX_STDOUT:
        for line in process.stdout.splitlines():
            Logger.info(line)


def main() -> None:
    if not _DIR_LATEX_FILES.exists():
        _DIR_LATEX_FILES.mkdir()

    Logger.info("Generating final LaTeX report")

    try:
        benchmark_obj: Benchmark = load_stats_models_from_json()
    except BenchmarkError as e:
        raise SystemExit(e) from e

    path_latex_source = _DIR_LATEX_FILES / "report.tex"
    latex_source = _assemble_full_text(benchmark_obj)
    path_latex_source.write_text(latex_source)

    try:
        _compile_latex_source(path_latex_source)
    except BenchmarkError as e:
        raise SystemExit(e) from e

    path_report_pdf = path_latex_source.with_suffix(".pdf")
    Logger.info("Exported final report to %s", path_report_pdf)


if __name__ == "__main__":
    main()
