REFERENCE_DEVICE = "target"

import importlib.util
import json
import sys
from pathlib import Path

import torch


_REFERENCE_PATH = Path(__file__).resolve().parent / "assets" / "reference.py"
_SPEC = importlib.util.spec_from_file_location("_vcd_reference_layer_norm_linear_fwd", _REFERENCE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load VCD reference asset: {_REFERENCE_PATH}")
_REFERENCE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _REFERENCE
_SPEC.loader.exec_module(_REFERENCE)
_OUTPUT_ARITY = 1


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


def _decode_vcd_value(value, tensors):
    if isinstance(value, dict):
        if "__tensor__" in value:
            return tensors[value["__tensor__"]]
        if "__dtype__" in value:
            return getattr(torch, value["__dtype__"].split(".")[-1])
        if "__tuple__" in value:
            return tuple(_decode_vcd_value(item, tensors) for item in value["__tuple__"])
        if "__list__" in value:
            return [_decode_vcd_value(item, tensors) for item in value["__list__"]]
        if "__dict__" in value:
            return {
                key: _decode_vcd_value(item, tensors)
                for key, item in value["__dict__"].items()
            }
    return value


def gen_inputs(ctx, device):
    del device
    from safetensors import safe_open

    path = ctx.get("input_path")
    if not isinstance(path, str):
        raise ValueError("VCD metadata input requires workload.input_path")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        scalars = json.loads(metadata.get("__scalars__", "{}"))
        nested_keys = json.loads(metadata.get("__nested_tensors__", "[]"))
        tensors = {key: handle.get_tensor(key) for key in nested_keys}
    return {
        key: _decode_vcd_value(value, tensors)
        for key, value in scalars.items()
        if key in ctx["inputs"]
    }


def run(x, norm_weight, norm_bias, linear_weight, linear_bias, eps):
    result = _REFERENCE.layer_norm_linear_fwd(x=x, norm_weight=norm_weight, norm_bias=norm_bias, linear_weight=linear_weight, linear_bias=linear_bias, eps=eps)
    return _project_tensor_outputs(result)
