.PHONY = clean py
.DEFAULT_GOAL = py

clean:
	@rm -rfv benchmark_inference_times/output benchmark_claude_skill_fix_memleak/build

py:
	@black benchmark_inference_times/src benchmark_claude_skill_fix_memleak/compile.py
	@pylint --exit-zero benchmark_inference_times/src benchmark_claude_skill_fix_memleak/compile.py
	@mypy --strict benchmark_inference_times/src benchmark_claude_skill_fix_memleak/compile.py
