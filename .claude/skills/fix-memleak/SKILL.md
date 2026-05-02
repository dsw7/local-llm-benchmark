---
description: Attempts to fix a memory leak in C/C++ if it exists.
---

Run the command:

! python3 run.py

This script will compile the source code under `src` and then run Valgrind against the
executable to check for memory leaks.

If a memory leak exists:
  The Valgrind `stdout` will list where the leak is coming from. Fix the memory leak.
Otherwise:
  Do nothing with the code and mention that no memory leak was found.
