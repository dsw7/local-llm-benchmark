# CLAUDE.md

## Project Overview

This is a Python utility for benchmarking locally hosted LLMs (via Ollama) across multiple servers/machines. It measures inference times for dummy prompts and generates statistical analysis, Gaussian distribution plots, and LaTeX reports.

**Primary purpose:** Benchmarking infrastructure (hardware, network, GPU models, VRAM) rather than model capabilities.

## Key Directories

- `commands/` - Python subcommands:
  - `subcommand_run_benchmarks.py` - Run inference benchmarks
  - `subcommand_plot_misc_statistics.py` - Generate statistics plots
  - `subcommand_generate_latex.py` - Generate LaTeX reports
- `commands/core/` - Core utilities (config loading, models, dataclass JSON IO)
- `configs.toml` - Production config (not tracked in git)
- `configs_example.toml` - Example configuration template
- `output/` - Generated plots and reports
- `docs/` - SVG example plots

## Key Files

- `benchmark` - Main bash script that provides menu options (1/2/3)
- `configs.toml` - Configuration (prompt, model name, rounds, servers)
- `configs_example.toml` - Example config template

## Configuration

The `configs.toml` file defines:
- `prompt` - Dummy prompt used for benchmarks
- `model` - Ollama model name (e.g., `gemma3:latest`)
- `rounds` - Number of inference measurements per host
- `[[servers]]` - List of hosts to benchmark (host, port)

## Running

1. Set up Python virtual environment
2. Run `./benchmark`
3. Choose option:
   - `1` - Run benchmarks (runs inference measurements)
   - `2` - Generate statistics (boxplots, distributions)
   - `3` - Generate LaTeX report

## Code Style

- Project uses Black for formatting
- Project uses pylint for linting
- Project uses mypy --strict for type checking
- Run `make py` to execute the CI pipeline

## Ollama Dependency

This project depends on Ollama running locally or on remote servers. The `model` config value must match a model loaded on the Ollama instance(s).

## Build Steps

To run the full CI pipeline:
```bash
make py
```

This runs:
1. `black commands` - Format code
2. `pylint --exit-zero commands` - Lint
3. `mypy --strict commands` - Type check
