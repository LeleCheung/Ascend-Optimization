REFERENCE_DEVICE = "target"

import importlib.util
import json
import sys
from pathlib import Path

import torch


_REFERENCE_PATH = Path(__file__).resolve().parent / "assets" / "reference.py"
_SPEC = importlib.util.spec_from_file_location("_vcd_reference_lora_shrink", _REFERENCE_PATH)
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
        if "__vllm_scalar_type__" in value:
            from vllm.scalar_type import scalar_types

            name = value["__vllm_scalar_type__"]
            if not isinstance(name, str) or name.startswith("_"):
                raise ValueError("invalid vLLM ScalarType metadata")
            result = getattr(scalar_types, name, None)
            if result is None:
                raise ValueError(f"unknown vLLM ScalarType: {name!r}")
            return result
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
        layouts = json.loads(metadata.get("__tensor_layouts__", "{}"))
        tensor_keys = set(nested_keys) | set(layouts)
        tensors = {key: handle.get_tensor(key) for key in tensor_keys}
    for key, layout in layouts.items():
        tensor = tensors.get(key)
        if tensor is None or not isinstance(layout, dict):
            raise ValueError("invalid VCD tensor layout metadata")
        size, stride = layout.get("size"), layout.get("stride")
        if (
            not isinstance(size, list)
            or not isinstance(stride, list)
            or len(size) != len(stride)
            or size != list(tensor.size())
        ):
            raise ValueError("invalid VCD tensor size or stride metadata")
        restored = torch.empty_strided(size, stride, dtype=tensor.dtype)
        restored.copy_(tensor)
        tensors[key] = restored
    overrides = {
        key: _decode_vcd_value(value, tensors)
        for key, value in scalars.items()
        if key in ctx["inputs"]
    }
    overrides.update(
        (key, tensors[key]) for key in layouts if key in ctx["inputs"]
    )
    return overrides


def _run(ns, inputs, lora_a_weights, num_tokens, rank, num_slices, dtype, token_lora_mapping, token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc, lora_ids, no_lora_flag_cpu, num_active_loras):
    output = torch.zeros(num_slices, num_tokens, rank, dtype=torch.float32, device='cuda')
    ns.lora_shrink(inputs, lora_a_weights, output, token_lora_mapping, token_indices_sorted_by_lora_ids, num_tokens_per_lora, lora_token_start_loc, lora_ids, no_lora_flag_cpu, num_active_loras, 1.0)
    return output

def _compute_ref_adapter(**kw):
    return _run(_REFERENCE, **kw)

def run(dtype, inputs, lora_a_weights, lora_ids, lora_token_start_loc, no_lora_flag_cpu, num_active_loras, num_slices, num_tokens, num_tokens_per_lora, rank, token_indices_sorted_by_lora_ids, token_lora_mapping):
    result = _compute_ref_adapter(dtype=dtype, inputs=inputs, lora_a_weights=lora_a_weights, lora_ids=lora_ids, lora_token_start_loc=lora_token_start_loc, no_lora_flag_cpu=no_lora_flag_cpu, num_active_loras=num_active_loras, num_slices=num_slices, num_tokens=num_tokens, num_tokens_per_lora=num_tokens_per_lora, rank=rank, token_indices_sorted_by_lora_ids=token_indices_sorted_by_lora_ids, token_lora_mapping=token_lora_mapping)
    return _project_tensor_outputs(result)
