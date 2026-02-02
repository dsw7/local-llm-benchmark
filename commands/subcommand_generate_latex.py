from subprocess import run, CalledProcessError
from core.consts import DIR_OUTPUT

_DIR_LATEX_FILES = DIR_OUTPUT / "latex"


def _assemble_technical_details_section() -> str:
    return r"""\section{Technical details}
This report was acquired on \today\ at \currenttime.
"""


def _assemble_full_text() -> str:
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

{_assemble_technical_details_section()}

\end{{document}}
"""


def main() -> None:
    if not _DIR_LATEX_FILES.exists():
        _DIR_LATEX_FILES.mkdir()

    path_report_tex = _DIR_LATEX_FILES / "report.tex"
    report = _assemble_full_text()
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
