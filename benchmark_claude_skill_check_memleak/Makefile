.PHONY = clean py
.DEFAULT_GOAL = py

clean:
	@rm -rfv build

py:
	@black run.py
	@pylint --exit-zero run.py
	@mypy --strict run.py
