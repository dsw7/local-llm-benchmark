# Local LLM benchmarking
Miscellenous utilities for benchmarking the use of locally hosted LLMs (i.e.
via [Ollama](https://ollama.com/)) with various tools and for various
platform/hardware permutations.

**Note that I am not interested in benchmarking the models themselves. I am
interested in benchmarking model inference times on my particular hardware.**
Many projects exist for benchmarking models themselves, such as
[SuperGLUE](https://super.gluebenchmark.com/).

I use this program to benchmark my infrastructure for the following cases:
- When using Claude Code with Ollama
- When running [FuncGraft](https://github.com/dsw7/FuncGraft) in [local
  mode](https://github.com/dsw7/FuncGraft?tab=readme-ov-file#toggling-between-llm-providers)
- When running [GPTifier](https://github.com/dsw7/GPTifier) commands via the Ollama stream

## Available suites
- [Benchmark inference times](#benchmark-inference-times)

## Benchmark inference times
- See [Benchmarking inference times](./benchmark_inference_times)

This suite runs a dummy prompt against a specified LLM on several machines and
several times. The inference times are gathered from which various basic
statistics are computed. This allows me to get a rough estimation of how
variables such as GPU models, available VRAM, etc., impact the overall
performance of my LLMs on prem.

The suite will generate Gaussian distributions for the various permutations
akin to:
<p align="center">
  <img width=600 src=./benchmark_inference_times/docs/example_gemma_3_n_50.svg>
</p>
In this example, 50 trials were performed. The mean inference time is around
2.15 seconds.
