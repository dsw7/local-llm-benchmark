.PHONY = clean py
.DEFAULT_GOAL = py

clean:
	@rm -rfv output

py:
	@black src
	@pylint --exit-zero src
	@mypy --strict src
