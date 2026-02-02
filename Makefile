.PHONY = clean py
.DEFAULT_GOAL = py

clean:
	@rm -rfv output

py:
	@black commands
	@pylint --exit-zero commands
	@mypy --strict commands
