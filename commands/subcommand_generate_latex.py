from subprocess import run, CalledProcessError
from core.dataclass_json_io import load_stats_models_from_json
from core.consts import DIR_OUTPUT
from core.exceptions import BenchmarkError

_DIR_LATEX_FILES = DIR_OUTPUT / "latex"


def _assemble_technical_details_section(prompt: str) -> str:
    return rf"""\section{{Technical details}}
\subsection*{{Basic test parameters:}}
\begin{{tabular}}{{|l|l|}}
  \textbf{{Parameter}} & \textbf{{Value}} \\
  \hline
  Acquisition date & \today \\
  \hline
  Acquisition time & \currenttime \\
\end{{tabular}}

\subsection*{{Prompt:}}
\begin{{verbatim}}
{prompt}
\end{{verbatim}}
\newpage
"""


def _assemble_full_text() -> str:
    _, prompt = load_stats_models_from_json()
    return rf"""\documentclass[10pt]{{article}}

\usepackage{{courier}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{titlesec}}
\usepackage{{datetime}}

\geometry{{margin=1in}}
\titleformat{{\section}}{{\bfseries\large}}{{}}{{0em}}{{}}[\titlerule]
\renewcommand*\familydefault{{\ttdefault}}

\begin{{document}}
\tableofcontents
\newpage

{_assemble_technical_details_section(prompt=prompt)}

\end{{document}}
"""


def main() -> None:
    if not _DIR_LATEX_FILES.exists():
        _DIR_LATEX_FILES.mkdir()

    try:
        report = _assemble_full_text()
    except BenchmarkError as e:
        raise SystemExit(e) from e

    path_report_tex = _DIR_LATEX_FILES / "report.tex"
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
