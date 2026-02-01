from subprocess import run, CalledProcessError
from core.consts import OutputDirectory, PlotsDirectory

LaTeXDirectory = OutputDirectory / "latex"


def _generate_preamble() -> str:
    return r"""\documentclass[10pt]{article}

\usepackage{courier}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{titlesec}
\usepackage{datetime}

\geometry{margin=1in}
\titleformat{\section}{\bfseries\large}{}{0em}{}[\titlerule]
\renewcommand*\familydefault{\ttdefault}
\graphicspath{{output/plots}}
"""


def _generate_body() -> str:
    return r"""
\begin{document}
\tableofcontents
\newpage

\section{Technical details}
This report was acquired on \today\ at \currenttime.

\end{document}
"""


def main() -> None:
    if not LaTeXDirectory.exists():
        LaTeXDirectory.mkdir()

    latex_file = LaTeXDirectory / "report.tex"
    markup = _generate_preamble() + _generate_body()
    latex_file.write_text(markup)

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
