from subprocess import run, CalledProcessError
from core.consts import OutputDirectory

LaTeXDirectory = OutputDirectory / "latex"


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
    if not LaTeXDirectory.exists():
        LaTeXDirectory.mkdir()

    latex_file = LaTeXDirectory / "report.tex"
    latex_file.write_text(_assemble_full_text())

    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        f"-output-directory={LaTeXDirectory}",
        f"{latex_file}",
    ]
    try:
        run(command, check=True)
    except CalledProcessError as e:
        raise SystemExit(e) from e


if __name__ == "__main__":
    main()
