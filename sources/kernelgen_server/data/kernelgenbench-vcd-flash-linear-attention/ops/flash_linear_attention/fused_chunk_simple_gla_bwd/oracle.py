REFERENCE_DEVICE = "target"

import importlib.util
import json
import sys
from pathlib import Path

import torch


_REFERENCE_PATH = Path(__file__).resolve().parent / "assets" / "reference.py"
_SPEC = importlib.util.spec_from_file_location("_vcd_reference_fused_chunk_simple_gla_bwd", _REFERENCE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load VCD reference asset: {_REFERENCE_PATH}")
_REFERENCE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _REFERENCE
_SPEC.loader.exec_module(_REFERENCE)
_OUTPUT_ARITY = 4


def _project_tensor_outputs(value):
    leaves = []

    def visit(item):
        if torch.is_tensor(item):
            leaves.append(item)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif item is not None and not isinstance(item, (bool, int, float, str)):
            raise TypeError(f"unsupported reference output leaf: {type(item).__name__}")

    visit(value)
    if not leaves:
        raise ValueError("reference produced no Tensor output")
    if len(leaves) > _OUTPUT_ARITY:
        raise ValueError(
            f"reference produced {len(leaves)} Tensor outputs; expected at most {_OUTPUT_ARITY}"
        )
    leaves.extend(
        torch.empty(0, dtype=torch.float32, device=leaves[0].device)
        for _ in range(_OUTPUT_ARITY - len(leaves))
    )
    return leaves[0] if len(leaves) == 1 else tuple(leaves)


def run(q, k, v, g, do):
    result = _REFERENCE.fused_chunk_simple_gla_bwd(q=q, k=k, v=v, g=g, do=do)
    return _project_tensor_outputs(result)
