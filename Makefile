.PHONY = clean py
.DEFAULT_GOAL = py

clean:
	@rm -rfv benchmark_inference_times/output

py:
	@black benchmark_inference_times/src
	@pylint --exit-zero benchmark_inference_times/src
	@mypy --strict benchmark_inference_times/src
