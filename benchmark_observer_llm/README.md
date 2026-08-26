# Benchmarking observer LLMs
This suite benchmarks a simple binary classifier. The classifier simply checks
whether an incoming prompt relates to editing code. This is achieved through a
combination of formatted user prompting, system prompting, and structured
outputs. The goal is to minimize the overhead associated with using an observer
LLM for moderation.

## Table of Contents
- [Setup](#setup)
- [Run the benchmarks](#run-the-benchmarks)

## Setup
Copy the example TOML file:
```bash
cp configs_example.toml configs.toml
```
The `configs.toml` file is the "production" file and is excluded via
`.gitignore`. Edit the file to match your specifications.

## Run the benchmarks
Set up a Python virtual environment and run the bash script:
```bash
./run
```
The program will attempt to classify a set of dummy instructions `rounds`
(specified via `configs.toml`) number of times against `model`. When complete,
the program will output something akin to:
```
* All values are provided in seconds
┌─────────────────┬───────────────┬─────────┬─────────┬──────────┬─────────┬─────────┬───────────────┐
│ Host            │ Model         │    Mean │      SD │   Median │     Min │     Max │   Sample size │
├─────────────────┼───────────────┼─────────┼─────────┼──────────┼─────────┼─────────┼───────────────┤
│ localhost:11434 │ gemma3:latest │ 1.03452 │ 0.02611 │  1.02438 │ 1.01501 │ 1.06418 │             5 │
└─────────────────┴───────────────┴─────────┴─────────┴──────────┴─────────┴─────────┴───────────────┘
```
