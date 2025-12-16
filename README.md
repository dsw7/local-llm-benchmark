# Local LLM benchmarking
Miscellenous utilities for benchmarking locally hosted LLMs for various platform/hardware permutations.

## Usage
Copy the example TOML file:
```bash
cp configs_example.toml configs.toml
```
The `configs.toml` file is the "production" file and is excluded via `.gitignore`. Set up a virtual environment
and run:
```bash
chmod +x benchmark.py && ./benchmark.py
```
