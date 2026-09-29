"""Pinned source-test policy; never a hardware probe or a FlagGems import."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
from importlib.resources import files
from copy import deepcopy
import json


_backend = ContextVar("native_source_backend", default=None)
_policy_id = ContextVar("native_source_policy_id", default=None)


class SourcePolicyError(ValueError):
    """Missing source policy is not a reason to select another oracle."""


def validate_selected_policy():
    if _policy_id.get() is not None:
        source_flag('support_fp64')


@lru_cache(maxsize=1)
def _config():
    return json.loads(files("kernelgen_server.runtime").joinpath("source_policy.json").read_text())


def policy_snapshot(backend):
    data = _config()
    vendor = data["backend_vendors"].get(backend)
    return deepcopy({"policy_id": data["policy_id"], "source": data["source"],
                     "vendor": vendor, "flags": data["vendors"].get(vendor)})


def source_flag(name):
    """Read a source flag inside a KGS-owned evaluation/profile scope."""
    data = _config()
    if data["policy_id"] != _policy_id.get():
        raise SourcePolicyError(f"unsupported or undeclared source policy: {_policy_id.get()}")
    flags = data["vendors"].get(data["backend_vendors"].get(_backend.get()))
    if flags is None or name not in flags:
        raise SourcePolicyError(f"unknown source policy flag {name!r} for backend {_backend.get()!r}")
    return flags[name]


def source_vendor():
    """Read the pinned vendor identity for source input-builder branches."""
    # Keep identity/unknown-backend checks identical to source_flag.
    source_flag('support_fp64')
    return _config()['backend_vendors'][_backend.get()]


@contextmanager
def source_policy_scope(backend, policy_id=None):
    token = _backend.set(backend)
    policy_token = _policy_id.set(policy_id)
    try:
        yield
    finally:
        _backend.reset(token)
        _policy_id.reset(policy_token)


def target_source_policy(method):
    """Bind the engine's logical backend, including CUDA-compatible vendors."""
    @wraps(method)
    def wrapped(self, request, *args, **kwargs):
        with source_policy_scope(self.backend, request.definition.source_policy_id):
            return method(self, request, *args, **kwargs)
    return wrapped


def skip_reason(workload):
    condition = workload.source_condition
    if condition is None:
        return None
    # Resolve all flags before deciding: an unknown condition is never a SKIP.
    flags = {name: source_flag(name)
             for name in condition.flags}
    unmet = [f"{name}={flags[name]} (requires {expected})"
             for name, expected in condition.flags.items() if flags[name] != expected]
    return "Source condition not met: " + ", ".join(unmet) if unmet else None
