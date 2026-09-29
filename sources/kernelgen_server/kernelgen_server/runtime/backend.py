"""Lightweight logical-backend addressing without importing accelerator runtimes."""

from __future__ import annotations


CUDA_COMPATIBLE_BACKENDS = frozenset(
    {"hygon", "metax", "iluvatar", "kunlunxin", "thead"}
)
KNOWN_BACKENDS = frozenset(
    {"cuda", "npu", "musa", "mlu", "enflame", "txda", *CUDA_COMPATIBLE_BACKENDS}
)


def runtime_device_type(backend: str) -> str:
    """Return the PyTorch device type used by a logical service backend."""

    if backend == "enflame":
        return "gcu"
    return "cuda" if backend in CUDA_COMPATIBLE_BACKENDS else backend


__all__ = [
    "CUDA_COMPATIBLE_BACKENDS",
    "KNOWN_BACKENDS",
    "runtime_device_type",
]
