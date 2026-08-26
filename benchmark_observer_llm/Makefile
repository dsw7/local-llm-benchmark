.PHONY = py

py:
	@black benchmark
	@pylint --exit-zero benchmark
	@mypy --strict benchmark
