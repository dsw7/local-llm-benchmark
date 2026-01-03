# Local LLM benchmarking
Miscellenous utilities for benchmarking locally hosted LLMs for various
platform/hardware permutations.

I use this program to benchmark my infrastructure for the following cases:
- When running [FuncGraft](https://github.com/dsw7/FuncGraft) in [local mode](https://github.com/dsw7/FuncGraft?tab=readme-ov-file#togging-between-llm-providers)

## Benchmarking LLM performance
### About
This script runs a dummy prompt against a specified LLM on several machines and
several times. The execution times are gathered from which the means and
standard deviations are calcaluted. I use this program mainly to fine tune
my hardware configurations.

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
┌──────────────────┬───────────────┬────────────┬──────────┬───────────────┐
│ Host             │ Model         │   Mean (s) │   SD (s) │   Sample size │
├──────────────────┼───────────────┼────────────┼──────────┼───────────────┤
│ localhost:11434  │ gemma3:latest │      2.19  │    0.036 │             3 │
│ 10.0.0.115:11434 │ gemma3:latest │     21.393 │    3.639 │             3 │
│ 10.0.0.243:11434 │ gemma3:latest │      8.455 │    0.024 │             3 │
└──────────────────┴───────────────┴────────────┴──────────┴───────────────┘
```
