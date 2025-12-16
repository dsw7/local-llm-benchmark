.PHONY = clean py
.DEFAULT_GOAL = py

py:
	@black benchmark.py
	@pylint --exit-zero benchmark.py
	@mypy --strict benchmark.py
