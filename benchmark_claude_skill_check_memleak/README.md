## Benchmark custom Claude memory leak check skill
This suite provides instructions for running a custom Claude skill that
troubleshoots C++ source with a known memory leak. The impetus is to determine
the amount of time it takes for Claude to understand the memory leak when
paired with a locally hosted LLM.

## Instructions
Ensure that Ollama is running in the background and then run:
```bash
claude --model=<the-model-to-test>
```
Then from the Claude shell run:
```
/clear
```
Followed by:
```
/explain-memleak
```
Not all models support tool calls. Ensure that you are testing a model that
supports tool calls.

> [!NOTE]
> You can run Claude from this directory, but it would be ideal to run Claude
> from the project's root directory (i.e. where `.claude` is located).
