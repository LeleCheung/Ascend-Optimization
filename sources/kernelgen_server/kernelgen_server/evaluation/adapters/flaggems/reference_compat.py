"""Narrow compatibility repairs for pinned framework reference callables."""

from __future__ import annotations

import importlib
from typing import Any


def prepare_flaggems_reference(operator: str) -> None:
    """Prepare one known-broken pinned reference before candidate import.

    The pinned MetaX ``linear_backward`` tune config passes the no-op
    ``SPLIT_K=1`` to a kernel that does not declare it. Keep this repair
    operator/vendor scoped so unrelated FlagGems baselines retain their native
    launch behavior.
    """

    if operator != "linear_backward":
        return
    flag_gems = importlib.import_module("flag_gems")
    if getattr(flag_gems, "vendor_name", "") != "metax":
        return
    _install_split_k_compat()


def _install_split_k_compat() -> None:
    jit = importlib.import_module("triton.runtime.jit")
    jit_function = jit.JITFunction
    if getattr(jit_function, "_kg_splitk_compat", False):
        return
    original = jit_function._pack_args

    def pack_args(self: Any, backend: Any, kwargs: dict[str, Any], *rest: Any):
        if kwargs.get("SPLIT_K") == 1:
            kwargs = {
                key: value for key, value in kwargs.items() if key != "SPLIT_K"
            }
        return original(self, backend, kwargs, *rest)

    pack_args._kg_splitk_compat = True
    jit_function._pack_args = pack_args
    jit_function._kg_splitk_compat = True


__all__ = ["prepare_flaggems_reference"]
