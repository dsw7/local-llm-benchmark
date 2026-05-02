---
name: explain-memleak
description: Checks if a program leaks memory and explains the problem is a memory leak is detected.
allowed-tools: Bash(python3 *)
---

# Context
! python3 benchmark_claude_skill_fix_memleak/run.py

# Your task
Explain how to fix the memory leak if a memory leak is found in the output. Do
not actually fix the memory leak. If no memory is found, then do nothing.
