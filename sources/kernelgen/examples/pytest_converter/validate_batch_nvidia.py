#!/usr/bin/env python3
"""Validate a reviewed conversion batch in the local NVIDIA container."""

try:
    from .validate_batch_metax import main
except ImportError:
    from validate_batch_metax import main


if __name__ == "__main__":
    main(default_local_nvidia=True)
