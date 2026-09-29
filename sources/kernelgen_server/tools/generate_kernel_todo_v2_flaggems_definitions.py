#!/usr/bin/env python3
"""Generate the manually grounded Kernel Todo V2 FlagGems Definitions.

The ordinary KernelGen extractor owns Definitions whose ABI can be read from a
public FlagGems callable.  This file covers two deterministic exceptions:

* operators whose FlagGems implementation intentionally exposes ``*args``;
* the 72 NVIDIA operators added by FlagGems commit 479844b2, whose candidate
  pytest uses ``resolve_gems_op`` before a public implementation is registered.

Every signature below is grounded in the pinned native ATen schema and the
matching correctness/benchmark calls.  The generator refuses to overwrite a
different existing Definition.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from kernelgen_server.protocol.schema import Definition


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS_ROOT = ROOT / "data" / "flaggems-adapter-definitions" / "definitions"


SIGNATURES = {
    # Existing FlagGems pytest whose public wrapper does not expose one
    # deterministic inspect.signature contract.
    "add_relu": "def run(self: Tensor, other: Tensor, *, alpha: Scalar = 1): pass",
    "asinh_": "def run(self: Tensor): pass",
    "functional_assert_async": (
        "def run(self: Tensor, assert_msg: str, dep_token: Tensor): pass"
    ),
    "ixor": "def run(self: Tensor, other): pass",
    "lift_fresh": "def run(self: Tensor): pass",
    "mvlgamma": "def run(self: Tensor, p: int): pass",
    "prelu_kernel_backward": (
        "def run(grad_output: Tensor, self: Tensor, weight: Tensor): pass"
    ),
    "rshift": "def run(self: Tensor, other): pass",
    "upsample_nearest_exact1d": (
        "def run(self: Tensor, output_size: List[int], "
        "scales: Optional[float] = None): pass"
    ),
    # NVIDIA Kernel Todo V2 operators added by FlagGems commit 479844b2.
    "_add_batch_dim": (
        "def run(self: Tensor, batch_dim: int, level: int): pass"
    ),
    "_choose_qparams_per_tensor": (
        "def run(self: Tensor, reduce_range: bool = False): pass"
    ),
    "_coalesce": "def run(self: Tensor): pass",
    "_dimI": "def run(self: Tensor): pass",
    "_dimV": "def run(self: Tensor): pass",
    "_dim_arange": "def run(like: Tensor, dim: int): pass",
    "_efficientzerotensor": (
        "def run(size: List[int], *, dtype: Optional[dtype] = None, "
        "layout: Optional[layout] = None, device: Optional[device] = None, "
        "pin_memory: Optional[bool] = None): pass"
    ),
    "_empty_affine_quantized": (
        "def run(size: List[int], *, dtype: Optional[dtype] = None, "
        "layout: Optional[layout] = None, device: Optional[device] = None, "
        "pin_memory: Optional[bool] = None, scale: float = 1.0, "
        "zero_point: int = 0, "
        "memory_format: Optional[memory_format] = 'contiguous_format'): pass"
    ),
    "_empty_per_channel_affine_quantized": (
        "def run(size: List[int], *, scales: Tensor, zero_points: Tensor, "
        "axis: int, dtype: Optional[dtype] = None, "
        "layout: Optional[layout] = None, device: Optional[device] = None, "
        "pin_memory: Optional[bool] = None, "
        "memory_format: Optional[memory_format] = 'contiguous_format'): pass"
    ),
    "_fw_primal": "def run(self: Tensor, level: int): pass",
    "_fw_primal_copy": "def run(self: Tensor, level: int): pass",
    "_has_same_storage_numel": (
        "def run(self: Tensor, other: Tensor): pass"
    ),
    "_indices": "def run(self: Tensor): pass",
    "_indices_copy": "def run(self: Tensor): pass",
    "_make_dual": (
        "def run(primal: Tensor, tangent: Tensor, level: int): pass"
    ),
    "_make_dual_copy": (
        "def run(primal: Tensor, tangent: Tensor, level: int): pass"
    ),
    "_make_per_channel_quantized_tensor": (
        "def run(self: Tensor, scale: Tensor, zero_point: Tensor, axis: int): pass"
    ),
    "_make_per_tensor_quantized_tensor": (
        "def run(self: Tensor, scale: float, zero_point: int): pass"
    ),
    "_neg_view": "def run(self: Tensor): pass",
    "_neg_view_copy": "def run(self: Tensor): pass",
    "_nested_compute_contiguous_strides_offsets": (
        "def run(nested_size: Tensor): pass"
    ),
    "_nested_tensor_size": "def run(self: Tensor): pass",
    "_nested_tensor_storage_offsets": "def run(self: Tensor): pass",
    "_nested_tensor_strides": "def run(self: Tensor): pass",
    "_new_zeros_with_same_feature_meta": (
        "def run(self: Tensor, other: Tensor, *, "
        "self_num_batch_dims: int = 0): pass"
    ),
    "_nnz": "def run(self: Tensor): pass",
    "_remove_batch_dim": (
        "def run(self: Tensor, level: int, batch_size: int, out_dim: int): pass"
    ),
    "_shape_as_tensor": "def run(self: Tensor): pass",
    "_slow_conv2d_backward": (
        "def run(grad_output: Tensor, self: Tensor, weight: Tensor, "
        "kernel_size: List[int], stride: List[int], padding: List[int], "
        "output_mask: List[bool]): pass"
    ),
    "_slow_conv2d_forward": (
        "def run(self: Tensor, weight: Tensor, kernel_size: List[int], "
        "bias: Optional[Tensor], stride: List[int], padding: List[int]): pass"
    ),
    "_unpack_dual": "def run(dual: Tensor, level: int): pass",
    "_values": "def run(self: Tensor): pass",
    "_values_copy": "def run(self: Tensor): pass",
    "_version": "def run(self: Tensor): pass",
    "adjoint": "def run(self: Tensor): pass",
    "atleast_1d": "def run(self: Tensor): pass",
    "atleast_2d": "def run(self: Tensor): pass",
    "atleast_3d": "def run(self: Tensor): pass",
    "can_cast": "def run(from_: dtype, to: dtype): pass",
    "cartesian_prod": "def run(tensors: List[Tensor]): pass",
    "ccol_indices": "def run(self: Tensor): pass",
    "ccol_indices_copy": "def run(self: Tensor): pass",
    "chain_matmul": "def run(matrices: List[Tensor]): pass",
    "coalesce": "def run(self: Tensor): pass",
    "col_indices": "def run(self: Tensor): pass",
    "col_indices_copy": "def run(self: Tensor): pass",
    "combinations": (
        "def run(self: Tensor, r: int = 2, "
        "with_replacement: bool = False): pass"
    ),
    "copy_sparse_to_sparse_": (
        "def run(self: Tensor, src: Tensor, non_blocking: bool = False): pass"
    ),
    "crow_indices": "def run(self: Tensor): pass",
    "crow_indices_copy": "def run(self: Tensor): pass",
    "data": "def run(self: Tensor): pass",
    "dense_dim": "def run(self: Tensor): pass",
    "detach_copy": "def run(self: Tensor): pass",
    "diagflat": "def run(self: Tensor, offset: int = 0): pass",
    "dim": "def run(self: Tensor): pass",
    "dstack": "def run(tensors: List[Tensor]): pass",
    "flatten_dense_tensors": "def run(tensors: List[Tensor]): pass",
    "slow_conv_dilated2d": (
        "def run(self: Tensor, weight: Tensor, kernel_size: List[int], "
        "bias: Optional[Tensor] = None, stride = 1, padding = 0, "
        "dilation = 1): pass"
    ),
    "slow_conv_dilated3d": (
        "def run(self: Tensor, weight: Tensor, kernel_size: List[int], "
        "bias: Optional[Tensor] = None, stride = 1, padding = 0, "
        "dilation = 1): pass"
    ),
    "slow_conv_transpose2d": (
        "def run(self: Tensor, weight: Tensor, kernel_size: List[int], "
        "bias: Optional[Tensor] = None, stride = 1, padding = 0, "
        "output_padding = 0, dilation = 1): pass"
    ),
    "slow_conv_transpose3d": (
        "def run(self: Tensor, weight: Tensor, kernel_size: List[int], "
        "bias: Optional[Tensor] = None, stride = 1, padding = 0, "
        "output_padding = 0, dilation = 1): pass"
    ),
    "sparse_bsc_tensor": (
        "def run(ccol_indices: Tensor, row_indices: Tensor, values: Tensor, "
        "size: Optional[List[int]] = None, *, dtype: Optional[dtype] = None, "
        "layout: Optional[layout] = None, device: Optional[device] = None, "
        "pin_memory: Optional[bool] = False): pass"
    ),
    "sparse_bsr_tensor": (
        "def run(crow_indices: Tensor, col_indices: Tensor, values: Tensor, "
        "size: Optional[List[int]] = None, *, dtype: Optional[dtype] = None, "
        "layout: Optional[layout] = None, device: Optional[device] = None, "
        "pin_memory: Optional[bool] = False): pass"
    ),
    "sparse_compressed_tensor": (
        "def run(compressed_indices: Tensor, plain_indices: Tensor, "
        "values: Tensor, size: Optional[List[int]] = None, *, "
        "dtype: Optional[dtype] = None, layout: Optional[layout] = None, "
        "device: Optional[device] = None, "
        "pin_memory: Optional[bool] = False): pass"
    ),
    "sparse_coo_tensor": (
        "def run(*args, dtype: Optional[dtype] = None, "
        "layout: Optional[layout] = None, device: Optional[device] = None, "
        "pin_memory: Optional[bool] = None, "
        "is_coalesced: Optional[bool] = None): pass"
    ),
    "sparse_csc_tensor": (
        "def run(ccol_indices: Tensor, row_indices: Tensor, values: Tensor, "
        "size: Optional[List[int]] = None, *, dtype: Optional[dtype] = None, "
        "layout: Optional[layout] = None, device: Optional[device] = None, "
        "pin_memory: Optional[bool] = False): pass"
    ),
    "sparse_csr_tensor": (
        "def run(crow_indices: Tensor, col_indices: Tensor, values: Tensor, "
        "size: Optional[List[int]] = None, *, dtype: Optional[dtype] = None, "
        "layout: Optional[layout] = None, device: Optional[device] = None, "
        "pin_memory: Optional[bool] = False): pass"
    ),
    "sparse_dim": "def run(self: Tensor): pass",
    "sparse_mask": "def run(self: Tensor, mask: Tensor): pass",
    "sparse_resize_": (
        "def run(self: Tensor, size: List[int], sparse_dim: int, "
        "dense_dim: int): pass"
    ),
    "sparse_resize_and_clear_": (
        "def run(self: Tensor, size: List[int], sparse_dim: int, "
        "dense_dim: int): pass"
    ),
    "thnn_conv2d": (
        "def run(self: Tensor, weight: Tensor, kernel_size: List[int], "
        "bias: Optional[Tensor] = None, stride = 1, padding = 0): pass"
    ),
}


OUTPUTS = {
    "_choose_qparams_per_tensor": ["scale", "zero_point"],
    "_nested_compute_contiguous_strides_offsets": ["strides", "offsets"],
    "_slow_conv2d_backward": ["grad_input", "grad_weight", "grad_bias"],
    "_unpack_dual": ["primal", "tangent"],
    "prelu_kernel_backward": ["grad_input", "grad_weight"],
}


MUTATES = {
    "asinh_": ["self"],
    "copy_sparse_to_sparse_": ["self"],
    "ixor": ["self"],
    "sparse_resize_": ["self"],
    "sparse_resize_and_clear_": ["self"],
}


SCHEMA_NAMES = {
    "add_relu": "_add_relu.Tensor",
    "functional_assert_async": "_functional_assert_async.msg",
    "ixor": "__ixor__.Tensor/Scalar",
    "prelu_kernel_backward": "_prelu_kernel_backward",
    "rshift": "__rshift__.Tensor/Scalar",
    "upsample_nearest_exact1d": "_upsample_nearest_exact1d",
}


def _literal(node: ast.expr) -> Any:
    return ast.literal_eval(node)


def _parameters(source: str) -> list[dict[str, Any]]:
    function = ast.parse(source).body[0]
    if not isinstance(function, ast.FunctionDef):
        raise TypeError("signature source must contain one function")
    arguments = function.args
    if arguments.kwarg is not None:
        raise ValueError("Definition does not support **kwargs")

    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(arguments.defaults)
    ) + list(arguments.defaults)
    result = []
    for index, (argument, default) in enumerate(zip(positional, defaults, strict=True)):
        parameter: dict[str, Any] = {
            "name": argument.arg,
            "kind": (
                "positional_only"
                if index < len(arguments.posonlyargs)
                else "positional_or_keyword"
            ),
            "required": default is None,
        }
        if argument.annotation is not None:
            parameter["type_hint"] = ast.unparse(argument.annotation)
        if default is not None:
            parameter["default"] = _literal(default)
        result.append(parameter)

    if arguments.vararg is not None:
        parameter = {
            "name": arguments.vararg.arg,
            "kind": "var_positional",
            "required": True,
        }
        if arguments.vararg.annotation is not None:
            parameter["type_hint"] = ast.unparse(arguments.vararg.annotation)
        result.append(parameter)

    for argument, default in zip(
        arguments.kwonlyargs, arguments.kw_defaults, strict=True
    ):
        parameter = {
            "name": argument.arg,
            "kind": "keyword_only",
            "required": default is None,
        }
        if argument.annotation is not None:
            parameter["type_hint"] = ast.unparse(argument.annotation)
        if default is not None:
            parameter["default"] = _literal(default)
        result.append(parameter)
    return result


def _payload(name: str, source: str) -> dict[str, Any]:
    outputs = OUTPUTS.get(name, ["out"])
    mutates = MUTATES.get(name, [])
    aliases = {outputs[0]: mutates[0]} if len(outputs) == 1 and mutates else {}
    schema_name = SCHEMA_NAMES.get(name, name)
    return {
        "name": name,
        "description": f"FlagGems adapter contract for aten::{schema_name}.",
        "parameters": _parameters(source),
        "outputs": outputs,
        "effects": {
            "mutates": mutates,
            "returns_alias_of": aliases,
        },
        "api_version": "v6.0",
    }


def main() -> None:
    if len(SIGNATURES) != 81:
        raise RuntimeError(f"expected 81 grounded signatures, got {len(SIGNATURES)}")
    DEFINITIONS_ROOT.mkdir(parents=True, exist_ok=True)
    written = 0
    unchanged = 0
    for name, source in sorted(SIGNATURES.items()):
        payload = _payload(name, source)
        Definition.model_validate(payload)
        path = DEFINITIONS_ROOT / f"{name}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise RuntimeError(f"refusing to overwrite different Definition: {path}")
            unchanged += 1
            continue
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written += 1
    print(f"wrote {written} Definitions; {unchanged} already matched")
    print(
        "deferred upsample_nearest_exact2d: one marker mixes default/out "
        "contracts that V6 Definition effects cannot express conditionally"
    )


if __name__ == "__main__":
    main()
