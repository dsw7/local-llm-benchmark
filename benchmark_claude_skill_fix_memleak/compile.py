#!/usr/bin/env python3

import sys
from pathlib import Path
from subprocess import run, CalledProcessError


def compile_source() -> Path:
    build_dir = Path("build")

    if not build_dir.exists():
        build_dir.mkdir()

    path_exec = build_dir / "main.out"
    command = ["g++", "-g", f"--output={path_exec}"]

    for p in Path("src").iterdir():
        if p.suffix == ".cpp":
            command.append(str(p))

    try:
        run(command, check=True)
    except CalledProcessError as e:
        sys.exit(str(e))

    return path_exec


def run_mem_check(path_exec: Path, leak_memory: bool) -> None:
    command = ["valgrind", "--leak-check=full", f"./{path_exec}"]

    if leak_memory:
        command.append("leak")
    else:
        command.append("no-leak")

    try:
        run(command, check=True)
    except CalledProcessError as e:
        sys.exit(str(e))


def main() -> None:
    path_exec = compile_source()
    run_mem_check(path_exec)


if __name__ == "__main__":
    main()
