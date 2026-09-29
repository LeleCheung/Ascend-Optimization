#!/usr/bin/env python3
"""Run reviewed pytest conversions on free remote T-Head PPU devices."""

from run_nvidia_backlog import main


if __name__ == "__main__":
    main(default_backend="remote-ppu")
