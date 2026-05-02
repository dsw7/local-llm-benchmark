---
name: explain-memleak
description: Checks if a program leaks memory and explains the problem is a memory leak is detected.
---

When asked to check for memory leaks, run the following steps:

# Run test script
Run the following:
```bash
python3 <project-root>/benchmark_claude_skill_fix_memleak/run.py
```

# Explain the output
If a memory leak is found, then explain the problem, but do not fix anything.
