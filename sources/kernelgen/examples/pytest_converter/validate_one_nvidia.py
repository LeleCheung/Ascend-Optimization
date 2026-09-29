#!/usr/bin/env python3
"""Validate one active pytest conversion in the local NVIDIA container."""

try:
    from .validate_one_metax import main
except ImportError:
    from validate_one_metax import main


if __name__ == "__main__":
    main(default_local_nvidia=True)
