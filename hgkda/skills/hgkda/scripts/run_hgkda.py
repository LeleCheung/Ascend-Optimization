#!/usr/bin/env python3
"""Invoke the installed HG-KDA core without changing interpreter paths or automatically installing dependencies."""
from hgkda.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
