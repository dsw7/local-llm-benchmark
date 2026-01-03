# Local LLM benchmarking
Miscellenous utilities for benchmarking locally hosted LLMs for various
platform/hardware permutations.

I use this program to benchmark my infrastructure for the following cases:
- When running [FuncGraft](https://github.com/dsw7/FuncGraft) in [local
  mode](https://github.com/dsw7/FuncGraft?tab=readme-ov-file#toggling-between-llm-providers)

## Benchmarking LLM performance
### About
This script runs a dummy prompt against a specified LLM on several machines and
several times. The execution times are gathered from which various basic
statistics are computed. This allows me to get a rough estimation of how
variables such as GPU models, available VRAM, etc., impact the overall
performance of my LLMs on prem.

### Usage
Copy the example TOML file:
```bash
cp configs_example.toml configs.toml
```
The `configs.toml` file is the "production" file and is excluded via
`.gitignore`. Edit the file to match your specifications (i.e. set the dummy
prompt and IP addresses). Then set up a virtual environment and run:
```bash
chmod +x benchmark.py && ./benchmark.py
```
When complete, the program will output something akin to:
```
All values are provided in seconds
┌──────────────────┬───────────────┬──────────┬─────────┬──────────┬──────────┬──────────┬───────────────┐
│ Host             │ Model         │     Mean │      SD │   Median │      Min │      Max │   Sample size │
├──────────────────┼───────────────┼──────────┼─────────┼──────────┼──────────┼──────────┼───────────────┤
│ localhost:11434  │ gemma3:latest │  2.18015 │ 0.16028 │  2.10775 │  2.09112 │  2.46496 │             5 │
│ 10.0.0.115:11434 │ gemma3:latest │ 18.0551  │ 0.62221 │ 17.9943  │ 17.3745  │ 19.0215  │             5 │
└──────────────────┴───────────────┴──────────┴─────────┴──────────┴──────────┴──────────┴───────────────┘
```
