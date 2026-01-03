# Local LLM benchmarking
Miscellenous utilities for benchmarking locally hosted LLMs for various
platform/hardware permutations.

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
