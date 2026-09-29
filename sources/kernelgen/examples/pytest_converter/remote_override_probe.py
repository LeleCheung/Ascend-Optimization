"""Pytest plugin proving that selected tests execute an override candidate."""

from __future__ import annotations

import json
import os
from pathlib import Path

import flag_gems


OPERATOR = os.environ["KERNELGEN_OVERRIDE_OPERATOR"]
OUTPUT = Path(os.environ["KERNELGEN_OVERRIDE_OUTPUT"])
CALLS: list[dict[str, object]] = []
DEFAULTS: list[object] = []
CONTEXT = None
ORIGINAL_RESOLVE = None


def _candidate(*args, **kwargs):
    if not DEFAULTS:
        raise RuntimeError("override candidate ran before resolve_gems_op")
    CALLS.append(
        {
            "positional_count": len(args),
            "keyword_names": sorted(kwargs),
        }
    )
    return DEFAULTS[-1](*args, **kwargs)


def pytest_configure(config):
    del config
    global CONTEXT
    global ORIGINAL_RESOLVE

    ORIGINAL_RESOLVE = flag_gems.testing.resolve_gems_op

    def tracking_resolve(name, default):
        if name == OPERATOR:
            DEFAULTS.append(default)
        return ORIGINAL_RESOLVE(name, default)

    flag_gems.testing.resolve_gems_op = tracking_resolve
    CONTEXT = flag_gems.testing.override_gems_op(OPERATOR, _candidate)
    CONTEXT.__enter__()


def pytest_unconfigure(config):
    del config
    restored = False
    error = None
    try:
        if CONTEXT is not None:
            CONTEXT.__exit__(None, None, None)
        if ORIGINAL_RESOLVE is not None:
            flag_gems.testing.resolve_gems_op = ORIGINAL_RESOLVE
        if DEFAULTS and ORIGINAL_RESOLVE is not None:
            restored = ORIGINAL_RESOLVE(OPERATOR, DEFAULTS[-1]) is DEFAULTS[-1]
    except Exception as exc:  # pragma: no cover - captured on the device
        error = f"{type(exc).__name__}: {exc}"
    OUTPUT.write_text(
        json.dumps(
            {
                "operator": OPERATOR,
                "call_count": len(CALLS),
                "calls": CALLS,
                "restored": restored,
                "error": error,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
