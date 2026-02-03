# Local LLM benchmarking
Miscellenous utilities for benchmarking locally hosted LLMs (i.e. via
[Ollama](https://ollama.com/)) for various platform/hardware permutations.

I use this program to benchmark my infrastructure for the following cases:
- When running [FuncGraft](https://github.com/dsw7/FuncGraft) in [local
  mode](https://github.com/dsw7/FuncGraft?tab=readme-ov-file#toggling-between-llm-providers)
- When running [GPTifier](https://github.com/dsw7/GPTifier) commands via the Ollama stream

## Table of Contents
- [About](#about)
- [Setup](#setup)
- [Benchmarking LLM performance](#benchmarking-llm-performance)
  - [Step 1 - Run the benchmarks](#step-1---run-the-benchmarks)
  - [Step 2 - Generate Gaussian distributions for inference times](#step-2---generate-gaussian-distributions-for-inference-times)
  - [Step 3 - Generate a LaTeX report for the measurements](#step-3---generate-a-latex-report-for-the-measurements)

## About
This programs runs a dummy prompt against a specified LLM on several machines
and several times. The execution times are gathered from which various basic
statistics are computed. This allows me to get a rough estimation of how
variables such as GPU models, available VRAM, etc., impact the overall
performance of my LLMs on prem.

## Setup
Copy the example TOML file:
```bash
cp configs_example.toml configs.toml
```
The `configs.toml` file is the "production" file and is excluded via
`.gitignore`. Edit the file to match your specifications (i.e. set the dummy
prompt and IP addresses).

## Benchmarking LLM performance

### Step 1 - Run the benchmarks
Set up a Python virtual environment and run the bash script:
```bash
./benchmark
```
And input <kbd>1</kbd> when prompted. The program will gather `rounds`
(specified via `configs.toml`) number of inference times for `prompt` against
`model` for each `host`. When complete, the program will output something akin
to:
```
All values are provided in seconds
┌──────────────────┬───────────────┬──────────┬─────────┬──────────┬──────────┬──────────┬───────────────┐
│ Host             │ Model         │     Mean │      SD │   Median │      Min │      Max │   Sample size │
├──────────────────┼───────────────┼──────────┼─────────┼──────────┼──────────┼──────────┼───────────────┤
│ localhost:11434  │ gemma3:latest │  2.18015 │ 0.16028 │  2.10775 │  2.09112 │  2.46496 │             5 │
│ 10.0.0.115:11434 │ gemma3:latest │ 18.0551  │ 0.62221 │ 17.9943  │ 17.3745  │ 19.0215  │             5 │
└──────────────────┴───────────────┴──────────┴─────────┴──────────┴──────────┴──────────┴───────────────┘
```
If sufficient, one can stop here.

### Step 2 - Generate Gaussian distributions for inference times
Set up a Python virtual environment as before and run the bash script:
```bash
./benchmark
```
Then input <kbd>2</kbd> when prompted. The program will generate a set of
Gaussian distributions for the inference times obtained from each machine. For
example:

<p align="center">
  <img width=600 src=./docs/example_gemma_3_n_50.svg>
</p>

In this example, 50 trials were performed. The mean inference time is around
2.15 seconds. One value appears to be more than 3 standard deviations away from
the mean, and this value could be interpreted as an outlier (perhaps as a
result of a spike in GPU demand).

### Step 3 - Generate a LaTeX report for the measurements
As before, run the bash script:
```bash
./benchmark
```
Then input <kbd>3</kbd> when prompted. The program will generate a full,
comprehensive report of all the statistics gathered as part of the benchmarking
process. Note that this requires that steps 1 and 2 be previously completed.
