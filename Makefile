.PHONY = clean py
.DEFAULT_GOAL = py

py:
	@black benchmark.py core/*.py
	@pylint --exit-zero benchmark.py core/*.py
	@mypy --strict benchmark.py core/*.py
