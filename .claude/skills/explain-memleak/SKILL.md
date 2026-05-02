---
name: explain-memleak
description: Checks if a program leaks memory and explains how to fix the problem is a memory leak is detected.
allowed-tools: Bash(python3 *) Bash(cd *)
---

# Context
! python3 benchmark_claude_skill_check_memleak/run.py --leak-memory

# Your task
Explain how to fix the memory leak if a memory leak is found in the output. Do
not actually fix the memory leak. If no memory leak is found, then do nothing.
