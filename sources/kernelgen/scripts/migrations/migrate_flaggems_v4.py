#!/usr/bin/env python3
"""Migrate the FlagGems v2 trace set to self-describing v4 workloads.

The v2 trace set stores each workload's tensor metadata in a reference-level
``CASES`` table selected by an opaque ``case_id`` axis.  V4 removes both:

* every tensor shape and dtype is written directly on its workload input;
* ordinary tensors use ``type: random`` and need no custom generator;
* constrained inputs keep ``type: custom`` and use a generator that reads only
  workload metadata plus existing scalar/literal inputs.

The source trace set is never modified.  The destination must not exist.
Run this script in an environment with torch because the source ``CASES`` tables
contain torch dtype objects.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_TAG = "schema:workload-shape-dtype-v4"


# Inputs that still require construction beyond the server's standard random
# tensor generator. All other v2 custom tensors become workload-driven random
# inputs and their Definition loses custom_inputs_entrypoint entirely.
SPECIAL_CUSTOM_INPUTS: dict[str, set[str]] = {
    "flaggems_scaled_dot_product_attention": {"q", "k", "v"},
    "flaggems_scaled_dot_product_attention_backward": {
        "grad_output",
        "query",
        "key",
        "value",
    },
    "flaggems_index_copy_": {"index"},
    "flaggems_masked_scatter_": {"mask"},
    "flaggems_max_unpool2d": {"indices"},
    "flaggems_one_hot": {"tensor"},
    "flaggems_linalg_cholesky": {"A"},
    "flaggems_adaptive_max_pool3d_backward": {
        "grad_output",
        "self_input",
        "indices",
    },
    "flaggems_bucketize": {"boundaries"},
    "flaggems_fmod_": {"y"},
    "flaggems_log10_": {"x"},
    "flaggems_resolve_neg": {"x"},
    "flaggems_rsqrt": {"x"},
    "flaggems_special_chebyshev_polynomial_w": {"x"},
    "flaggems_special_gammainc": {"a", "x"},
    "flaggems_embedding_dense_backward": {"indices"},
    "flaggems_histc": {"x"},
    "flaggems_median": {"x"},
    "flaggems_nll_loss_backward": {
        "grad_output",
        "self_input",
        "target",
        "weight",
        "total_weight",
    },
    "flaggems_nonzero_numpy": {"x"},
    "flaggems_scatter_reduce_": {"index"},
    "flaggems_concatenate": {"tensors"},
    "flaggems_unique_consecutive": {"x"},
}


GENERATOR_SOURCES: dict[str, str] = {
    "flaggems_scaled_dot_product_attention": """
def gen_inputs(ctx, device):
    result = {}
    for name in ("q", "k", "v"):
        result[name] = torch.empty(
            ctx[f"{name}__shape"],
            dtype=_v4_dtype(ctx, name),
            device=device,
        ).uniform_(-0.05, 0.05)
    return result
""",
    "flaggems_scaled_dot_product_attention_backward": """
def gen_inputs(ctx, device):
    result = {}
    for name in ("query", "key", "value"):
        result[name] = torch.empty(
            ctx[f"{name}__shape"],
            dtype=_v4_dtype(ctx, name),
            device=device,
        ).uniform_(-0.05, 0.05)
    result["grad_output"] = torch.randn(
        ctx["grad_output__shape"],
        dtype=_v4_dtype(ctx, "grad_output"),
        device=device,
    )
    return result
""",
    "flaggems_index_copy_": """
def gen_inputs(ctx, device):
    return {
        "index": torch.randperm(
            ctx["index__shape"][0],
            dtype=_v4_dtype(ctx, "index"),
            device=device,
        )
    }
""",
    "flaggems_masked_scatter_": """
def gen_inputs(ctx, device):
    mask_shape = ctx["mask__shape"]
    source_numel = ctx["source__shape"][0]
    mask_numel = 1
    for extent in mask_shape:
        mask_numel *= extent
    mask = torch.arange(mask_numel, device=device) < source_numel
    return {"mask": mask.reshape(mask_shape)}
""",
    "flaggems_max_unpool2d": """
def gen_inputs(ctx, device):
    output_size = ctx["output_size"]
    upper = output_size[-2] * output_size[-1]
    return {
        "indices": torch.randint(
            0,
            upper,
            ctx["indices__shape"],
            dtype=_v4_dtype(ctx, "indices"),
            device=device,
        )
    }
""",
    "flaggems_one_hot": """
def gen_inputs(ctx, device):
    return {
        "tensor": torch.randint(
            0,
            ctx["num_classes"],
            ctx["tensor__shape"],
            dtype=_v4_dtype(ctx, "tensor"),
            device=device,
        )
    }
""",
    "flaggems_linalg_cholesky": """
def gen_inputs(ctx, device):
    shape = ctx["A__shape"]
    dtype = _v4_dtype(ctx, "A")
    base = torch.randn(shape, dtype=dtype, device=device)
    matrix = base @ base.transpose(-2, -1)
    matrix = matrix + torch.eye(shape[-1], dtype=dtype, device=device) * 0.1
    return {"A": matrix}
""",
    "flaggems_adaptive_max_pool3d_backward": """
def gen_inputs(ctx, device):
    self_shape = ctx["self_input__shape"]
    self_input = torch.randn(
        self_shape, dtype=_v4_dtype(ctx, "self_input"), device=device
    )
    output_size = ctx["indices__shape"][-3:]
    pooled, indices = torch.nn.functional.adaptive_max_pool3d(
        self_input, output_size=output_size, return_indices=True
    )
    return {
        "grad_output": torch.ones_like(pooled),
        "self_input": self_input,
        "indices": indices,
    }
""",
    "flaggems_bucketize": """
def gen_inputs(ctx, device):
    return {
        "boundaries": torch.tensor(
            [1.0, 3.0, 5.0, 7.0, 9.0],
            dtype=torch.float32,
            device=device,
        )
    }
""",
    "flaggems_fmod_": """
def gen_inputs(ctx, device):
    y = torch.randn(
        ctx["y__shape"], dtype=_v4_dtype(ctx, "y"), device=device
    )
    y = torch.where(y == 0, torch.ones_like(y), y)
    return {"y": y}
""",
    "flaggems_log10_": """
def gen_inputs(ctx, device):
    x = torch.rand(
        ctx["x__shape"], dtype=_v4_dtype(ctx, "x"), device=device
    ) + 0.1
    return {"x": x}
""",
    "flaggems_resolve_neg": """
def gen_inputs(ctx, device):
    base = torch.randn(ctx["x__shape"], dtype=torch.complex64).to(device=device)
    result = base.conj().imag
    assert result.is_neg()
    return {"x": result}
""",
    "flaggems_rsqrt": """
def gen_inputs(ctx, device):
    x = torch.rand(
        ctx["x__shape"], dtype=_v4_dtype(ctx, "x"), device=device
    ) * 10.0 + 0.1
    return {"x": x}
""",
    "flaggems_special_chebyshev_polynomial_w": """
def gen_inputs(ctx, device):
    x = torch.rand(
        ctx["x__shape"], dtype=_v4_dtype(ctx, "x"), device=device
    ) * 2.0 - 1.0
    return {"x": x}
""",
    "flaggems_special_gammainc": """
def gen_inputs(ctx, device):
    result = {}
    for name in ("a", "x"):
        result[name] = torch.rand(
            ctx[f"{name}__shape"],
            dtype=_v4_dtype(ctx, name),
            device=device,
        ) * 10.0 + 0.1
    return result
""",
    "flaggems_embedding_dense_backward": """
def gen_inputs(ctx, device):
    return {
        "indices": torch.randint(
            0,
            ctx["num_weights"],
            ctx["indices__shape"],
            dtype=_v4_dtype(ctx, "indices"),
            device=device,
        )
    }
""",
    "flaggems_histc": """
def gen_inputs(ctx, device):
    shape = ctx["x__shape"]
    dtype = _v4_dtype(ctx, "x")
    numel = 1
    for extent in shape:
        numel *= extent
    bucket_ids = torch.arange(numel, device=device) % 100
    values = (bucket_ids.to(dtype) + 0.5) * 0.1
    return {"x": values.reshape(shape)}
""",
    "flaggems_median": """
def gen_inputs(ctx, device):
    shape = ctx["x__shape"]
    dtype = _v4_dtype(ctx, "x")
    if dtype.is_floating_point:
        return {"x": torch.randn(shape, dtype=dtype, device=device)}
    numel = 1
    for extent in shape:
        numel *= extent
    values = torch.arange(numel, device=device, dtype=torch.int64)
    values = values * 37 % numel - numel // 2
    return {"x": values.reshape(shape).to(dtype)}
""",
    "flaggems_nll_loss_backward": """
def gen_inputs(ctx, device):
    self_shape = ctx["self_input__shape"]
    dtype = _v4_dtype(ctx, "self_input")
    classes = self_shape[1]
    target_shape = ctx["target__shape"]
    target = torch.randint(
        0, classes, target_shape, dtype=_v4_dtype(ctx, "target"), device=device
    )
    self_input = torch.randn(self_shape, dtype=dtype, device=device)
    if "weight__shape" in ctx:
        weight = torch.randn(
            ctx["weight__shape"], dtype=_v4_dtype(ctx, "weight"), device=device
        )
        selected_weight = weight[target]
    else:
        weight = None
        selected_weight = torch.ones(target_shape, dtype=dtype, device=device)
    mask = (target != ctx["ignore_index"]).to(dtype)
    total_weight = (selected_weight * mask).sum()
    grad_output = torch.randn(
        ctx["grad_output__shape"],
        dtype=_v4_dtype(ctx, "grad_output"),
        device=device,
    )
    result = {
        "grad_output": grad_output,
        "self_input": self_input,
        "target": target,
        "total_weight": total_weight,
    }
    if weight is not None:
        result["weight"] = weight
    return result
""",
    "flaggems_nonzero_numpy": """
def gen_inputs(ctx, device):
    shape = ctx["x__shape"]
    dtype = _v4_dtype(ctx, "x")
    if dtype == torch.bool:
        value = torch.randint(0, 2, shape, dtype=torch.int32, device=device).to(dtype)
    elif dtype in (torch.int16, torch.int32, torch.int64):
        value = torch.randint(-3, 3, shape, device=device).to(dtype)
    else:
        value = torch.randn(shape, dtype=dtype, device=device)
        value.reshape(-1)[::7] = 0
    return {"x": value}
""",
    "flaggems_scatter_reduce_": """
def gen_inputs(ctx, device):
    dim = ctx["dim"]
    input_shape = ctx["x__shape"]
    return {
        "index": torch.randint(
            0,
            input_shape[dim],
            ctx["index__shape"],
            dtype=_v4_dtype(ctx, "index"),
            device=device,
        )
    }
""",
    "flaggems_concatenate": """
def gen_inputs(ctx, device):
    dtype = _v4_dtype(ctx, "tensors")
    tensors = []
    for shape in ctx["tensors__shape"]:
        if dtype in (torch.float16, torch.float32, torch.bfloat16):
            value = torch.randn(shape, dtype=dtype, device=device)
        else:
            value = torch.randint(0, 32767, shape, dtype=dtype, device=device)
        tensors.append(value)
    return {"tensors": tensors}
""",
    "flaggems_unique_consecutive": """
def gen_inputs(ctx, device):
    value = torch.randint(-5, 5, ctx["x__shape"], device=device)
    return {"x": value.to(_v4_dtype(ctx, "x"))}
""",
}


GENERATOR_HELPER = """
def _v4_dtype(ctx, name):
    return getattr(torch, ctx[f"{name}__dtype"])
"""

REFERENCE_RUN_SOURCES: dict[str, str] = {
    "flaggems_bucketize": """
def run(x, boundaries):
    return torch.bucketize(x.to(torch.float32), boundaries)
""",
}

REFERENCE_VALID_SOURCES: dict[str, str] = {
    "flaggems_max_pool3d_with_indices": """
def valid(ref_outputs, sol_outputs, inputs, ctx):
    passed = torch.allclose(ref_outputs[0], sol_outputs[0])
    return {
        "passed": passed,
        "message": "OK" if passed else "Output values mismatch",
        "metrics": {},
    }
""",
}

PORTABLE_TIMING_DTYPES: dict[str, set[str]] = {
    # The original benchmark includes FP16 inputs, but MUSA has no native
    # torch.bucketize FP16 baseline. Keep only the cross-device comparable case.
    "flaggems_bucketize": {"float32"},
}


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _shape_list(shape: Any) -> list[Any]:
    if isinstance(shape, (list, tuple)):
        return [
            _shape_list(item) if isinstance(item, (list, tuple)) else int(item)
            for item in shape
        ]
    raise TypeError(f"Expected a shape sequence, got {shape!r}")


def _tensor(shape: Any, dtype: Any) -> dict[str, Any]:
    return {
        "type": "custom",
        "shape": _shape_list(shape),
        "dtype": _dtype_name(dtype),
    }


def _scalar(value: Any) -> dict[str, Any]:
    return {"type": "scalar", "value": value}


def _literal(value: Any) -> dict[str, Any]:
    return {"type": "literal", "value": value}


def _pool_output_dim(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


def _case_metadata(
    name: str,
    case: Any,
    workload_inputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Resolve the tensors/scalars produced by the v2 generator without allocation."""
    custom_names = {
        input_name
        for input_name, spec in workload_inputs.items()
        if spec.get("type") == "custom"
    }

    if name == "flaggems_scaled_dot_product_attention":
        batch, q_heads, kv_heads, q_seq, kv_seq, head_size, dtype = case
        return {
            "q": _tensor((batch, q_heads, q_seq, head_size), dtype),
            "k": _tensor((batch, kv_heads, kv_seq, head_size), dtype),
            "v": _tensor((batch, kv_heads, kv_seq, head_size), dtype),
        }
    if name == "flaggems_scaled_dot_product_attention_backward":
        batch, heads, q_seq, kv_seq, head_size, dtype, _ = case
        return {
            "grad_output": _tensor((batch, heads, q_seq, head_size), dtype),
            "query": _tensor((batch, heads, q_seq, head_size), dtype),
            "key": _tensor((batch, heads, kv_seq, head_size), dtype),
            "value": _tensor((batch, heads, kv_seq, head_size), dtype),
        }
    if name == "flaggems_reflection_pad3d_backward":
        shape, dtype, padding = case
        d0, d1, h0, h1, w0, w1 = padding
        n, channels, depth, height, width = shape
        padded = (
            n,
            channels,
            depth + d0 + d1,
            height + h0 + h1,
            width + w0 + w1,
        )
        return {
            "grad_output": _tensor(padded, dtype),
            "self_input": _tensor(shape, dtype),
        }
    if name in {"flaggems_cudnn_convolution", "flaggems_conv_transpose1d"}:
        input_shape, weight_shape, dtype = case
        input_name = "input" if name == "flaggems_cudnn_convolution" else "x"
        return {
            input_name: _tensor(input_shape, dtype),
            "weight": _tensor(weight_shape, dtype),
        }
    if name == "flaggems_addmm_":
        (rows, columns, inner), dtype = case
        return {
            "x": _tensor((rows, columns), dtype),
            "mat1": _tensor((rows, inner), dtype),
            "mat2": _tensor((inner, columns), dtype),
        }
    if name == "flaggems_linear":
        x_shape, weight_shape, bias_shape, dtype = case
        return {
            "x": _tensor(x_shape, dtype),
            "weight": _tensor(weight_shape, dtype),
            "bias": (
                _tensor(bias_shape, dtype)
                if bias_shape is not None
                else _literal(None)
            ),
        }
    if name == "flaggems_index_copy_":
        shape, dtype, dim = case
        return {
            "x": _tensor(shape, dtype),
            "dim": _scalar(dim),
            "index": _tensor((shape[dim],), "int64"),
            "src": _tensor(shape, dtype),
        }
    if name == "flaggems_masked_scatter_":
        shape, dtype, threshold = case
        numel = math.prod(shape)
        probability = 0.5 * (1.0 + math.erf(threshold / math.sqrt(2.0)))
        source_size = min(numel, max(1, round(numel * probability)))
        return {
            "inp": _tensor(shape, dtype),
            "mask": _tensor(shape, "bool"),
            "source": _tensor((source_size,), dtype),
        }
    if name == "flaggems_max_unpool2d":
        shape, (kernel, stride, padding), dtype = case
        pooled_shape = (
            shape[0],
            shape[1],
            _pool_output_dim(shape[2], kernel, stride, padding),
            _pool_output_dim(shape[3], kernel, stride, padding),
        )
        return {
            "pooled": _tensor(pooled_shape, dtype),
            "indices": _tensor(pooled_shape, "int64"),
        }
    if name == "flaggems_one_hot":
        shape, num_classes = case
        return {
            "tensor": _tensor(shape, "int64"),
            "num_classes": _scalar(num_classes),
        }
    if name == "flaggems_adaptive_max_pool3d_backward":
        shape, output_size, dtype = case
        result_shape = tuple(shape[:2]) + tuple(output_size)
        return {
            "grad_output": _tensor(result_shape, dtype),
            "self_input": _tensor(shape, dtype),
            "indices": _tensor(result_shape, "int32"),
        }
    if name == "flaggems_bucketize":
        shape, dtype = case
        return {
            "x": _tensor(shape, dtype),
            "boundaries": _tensor((5,), "float32"),
        }
    if name == "flaggems_resolve_neg":
        shape, _ = case
        return {"x": _tensor(shape, "float32")}
    if name == "flaggems_avg_pool3d_backward":
        shape, dtype = case
        kernel = workload_inputs["kernel_size"]["value"]
        stride = workload_inputs["stride"]["value"]
        padding = workload_inputs["padding"]["value"]
        result_shape = list(shape[:2])
        for size, k, s, p in zip(shape[2:], kernel, stride, padding):
            result_shape.append(_pool_output_dim(size, k, s, p))
        return {
            "grad_output": _tensor(result_shape, dtype),
            "self_input": _tensor(shape, dtype),
        }
    if name == "flaggems_embedding_dense_backward":
        (batch, length, width), num_weights, dtype = case
        return {
            "grad_output": _tensor((batch, length, width), dtype),
            "indices": _tensor((batch, length), "int64"),
        }
    if name == "flaggems_histc":
        shape, dtype, *_ = case
        return {"x": _tensor(shape, dtype)}
    if name == "flaggems_nll_loss_backward":
        shape, dtype, reduction, has_weight, _ = case
        target_shape = list(shape)
        del target_shape[1]
        return {
            "grad_output": _tensor(
                target_shape if reduction == 0 else (), dtype
            ),
            "self_input": _tensor(shape, dtype),
            "target": _tensor(target_shape, "int64"),
            "weight": (
                _tensor((shape[1],), dtype) if has_weight else _literal(None)
            ),
            "total_weight": _tensor((), dtype),
        }
    if name == "flaggems_upsample_linear1d_backward":
        shape, *_, dtype = case
        return {"grad_output": _tensor(shape, dtype)}
    if name == "flaggems_rnn_relu":
        seq_len, batch, input_size, hidden_size, dtype, batch_first = case
        input_shape = (
            (batch, seq_len, input_size)
            if batch_first
            else (seq_len, batch, input_size)
        )
        return {
            "input": _tensor(input_shape, dtype),
            "hx": _tensor((1, batch, hidden_size), dtype),
            "w_ih": _tensor((hidden_size, input_size), dtype),
            "w_hh": _tensor((hidden_size, hidden_size), dtype),
            "b_ih": _tensor((hidden_size,), dtype),
            "b_hh": _tensor((hidden_size,), dtype),
        }
    if name == "flaggems_scatter_reduce_":
        input_shape, source_shape, _, dtype = case
        return {
            "x": _tensor(input_shape, dtype),
            "index": _tensor(source_shape, "int64"),
            "src": _tensor(source_shape, dtype),
        }
    if name == "flaggems_broadcast_to":
        source_shape, _, dtype = case
        return {"x": _tensor(source_shape, dtype)}
    if name == "flaggems_concatenate":
        shapes, dtype = case
        return {"tensors": _tensor(shapes, dtype)}

    if not (
        isinstance(case, tuple)
        and len(case) == 2
        and isinstance(case[0], (tuple, list))
    ):
        raise ValueError(f"{name}: no metadata resolver for CASE {case!r}")
    shape, dtype = case
    return {input_name: _tensor(shape, dtype) for input_name in custom_names}


def _target_is_cases(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "CASES"
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_target_is_cases(item) for item in node.elts)
    return False


def _rewrite_reference(definition: dict[str, Any]) -> str:
    entrypoint = definition.get("custom_inputs_entrypoint") or "gen_inputs"
    run_override = REFERENCE_RUN_SOURCES.get(definition["name"])
    valid_override = REFERENCE_VALID_SOURCES.get(definition["name"])
    module = ast.parse(definition["reference"])
    body: list[ast.stmt] = []
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            _target_is_cases(target) for target in node.targets
        ):
            continue
        if isinstance(node, ast.AnnAssign) and _target_is_cases(node.target):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == entrypoint
        ):
            continue
        if (
            run_override
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ):
            continue
        if (
            valid_override
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "valid"
        ):
            continue
        body.append(node)
    module.body = body
    ast.fix_missing_locations(module)
    reference = ast.unparse(module).rstrip()

    if run_override:
        reference += "\n\n" + run_override.strip()
    if valid_override:
        reference += "\n\n" + valid_override.strip()
    generator = GENERATOR_SOURCES.get(definition["name"])
    if generator:
        if "_v4_dtype" in generator:
            reference += "\n\n" + GENERATOR_HELPER.strip()
        reference += "\n\n" + generator.strip()
    return reference + "\n"


def _migrate_definition(definition: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(definition)
    result["axes"].pop("case_id", None)
    if definition["name"] == "flaggems_adaptive_max_pool3d_backward":
        result["inputs"]["indices"]["dtype"] = "dynamic"
    if definition["name"] == "flaggems_bucketize":
        result["inputs"]["boundaries"]["dtype"] = "float32"
    result["reference"] = _rewrite_reference(definition)
    tags = [
        tag
        for tag in result.get("tags", [])
        if not tag.startswith("schema:") and tag != "eval:statistical"
    ]
    result["tags"] = [*tags, SCHEMA_TAG]
    reference_module = ast.parse(result["reference"])
    if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "valid"
        for node in reference_module.body
    ):
        result["custom_valid_entrypoint"] = "valid"
    else:
        result.pop("custom_valid_entrypoint", None)
    if definition["name"] in GENERATOR_SOURCES:
        result["custom_inputs_entrypoint"] = "gen_inputs"
    else:
        result.pop("custom_inputs_entrypoint", None)
    return result


def _migrate_workload(
    definition_name: str,
    cases: dict[int, Any],
    wrapped_trace: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(wrapped_trace)
    workload = result["workload"]
    case_id = workload.get("axes", {}).get("case_id")
    if case_id is None:
        raise ValueError(f"{definition_name}/{workload['uuid']}: missing v2 case_id")
    metadata = _case_metadata(definition_name, cases[case_id], workload["inputs"])
    special_names = SPECIAL_CUSTOM_INPUTS.get(definition_name, set())

    migrated_inputs: dict[str, dict[str, Any]] = {}
    for input_name, old_spec in workload["inputs"].items():
        if old_spec.get("type") != "custom":
            migrated_inputs[input_name] = copy.deepcopy(old_spec)
            continue
        if input_name not in metadata:
            raise ValueError(
                f"{definition_name}/{workload['uuid']}: no metadata for {input_name}"
            )
        new_spec = copy.deepcopy(metadata[input_name])
        if (
            new_spec["type"] == "custom"
            and input_name not in special_names
        ):
            new_spec["type"] = "random"
        migrated_inputs[input_name] = new_spec

    workload["axes"] = {
        key: value
        for key, value in workload.get("axes", {}).items()
        if key != "case_id"
    }
    workload["inputs"] = migrated_inputs
    return result


def _include_workload(
    definition_name: str,
    phase: str,
    wrapped_trace: dict[str, Any],
) -> bool:
    if phase != "timing":
        return True
    allowed_dtypes = PORTABLE_TIMING_DTYPES.get(definition_name)
    if allowed_dtypes is None:
        return True
    x_dtype = wrapped_trace["workload"]["inputs"]["x"].get("dtype")
    return x_dtype in allowed_dtypes


def _load_cases(reference: str) -> dict[int, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(reference, "<v2-reference>", "exec"), namespace)
    cases = namespace.get("CASES")
    if not isinstance(cases, dict):
        raise ValueError("v2 reference does not define a CASES dict")
    return cases


def migrate(source: Path, destination: Path) -> dict[str, int]:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    definitions_root = source / "definitions"
    workloads_root = source / "phased_workloads"
    if not definitions_root.is_dir() or not workloads_root.is_dir():
        raise FileNotFoundError(
            f"Expected definitions/ and phased_workloads/ under {source}"
        )

    counts = {"definitions": 0, "workloads": 0, "custom_inputs": 0}
    for definition_path in sorted(definitions_root.rglob("*.json")):
        definition = json.loads(definition_path.read_text())
        name = definition["name"]
        op_type = definition["op_type"]
        cases = _load_cases(definition["reference"])
        migrated_definition = _migrate_definition(definition)

        output_definition = destination / "definitions" / op_type / f"{name}.json"
        output_definition.parent.mkdir(parents=True, exist_ok=True)
        output_definition.write_text(
            json.dumps(migrated_definition, indent=2, ensure_ascii=False) + "\n"
        )
        counts["definitions"] += 1

        for phase in ("correctness", "timing"):
            input_workloads = (
                workloads_root / op_type / f"{name}.{phase}.jsonl"
            )
            output_workloads = (
                destination
                / "phased_workloads"
                / op_type
                / f"{name}.{phase}.jsonl"
            )
            output_workloads.parent.mkdir(parents=True, exist_ok=True)
            migrated_lines = []
            for line in input_workloads.read_text().splitlines():
                if not line.strip():
                    continue
                migrated = _migrate_workload(name, cases, json.loads(line))
                if not _include_workload(name, phase, migrated):
                    continue
                counts["workloads"] += 1
                counts["custom_inputs"] += sum(
                    spec.get("type") == "custom"
                    for spec in migrated["workload"]["inputs"].values()
                )
                migrated_lines.append(
                    json.dumps(migrated, separators=(",", ":"), ensure_ascii=False)
                )
            output_workloads.write_text("\n".join(migrated_lines) + "\n")

    if counts["definitions"] != 66:
        raise RuntimeError(
            f"Expected 66 definitions, migrated {counts['definitions']}"
        )

    forbidden = []
    for path in destination.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text()
        if "CASES" in text or '"case_id"' in text:
            forbidden.append(str(path))
    if forbidden:
        raise RuntimeError(
            "V4 output still contains CASES/case_id: " + ", ".join(forbidden)
        )

    readme = destination / "README.md"
    readme.write_text(
        "# unified-trace-flaggems-v4\n\n"
        "Migrated from v2 with workload-level shape/dtype metadata.\n\n"
        f"- Definitions: {counts['definitions']}\n"
        f"- Workloads: {counts['workloads']}\n"
        f"- Remaining custom tensor inputs: {counts['custom_inputs']}\n"
        "- Legacy reference lookup tables: 0\n"
        "- Legacy selector axes: 0\n"
        "- Compatibility baseline: `unified-trace-akg-bench-lite`\n"
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    counts = migrate(args.source.resolve(), args.destination.resolve())
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
