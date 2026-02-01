.PHONY = clean py
.DEFAULT_GOAL = py

py:
	@black commands
	@pylint --exit-zero commands
	@mypy --strict commands
