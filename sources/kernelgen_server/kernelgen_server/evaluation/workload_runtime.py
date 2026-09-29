"""Materialize V6 workloads and bind them to the operator ABI."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Mapping

from ..protocol.schema import Definition, ParameterKind, Workload
from .call import Call
from .pytree import clone


def _torch_dtype(dtype_name: str) -> Any:
    import torch

    value = getattr(torch, dtype_name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported random dtype: {dtype_name}")
    return value


def _random_tensor(
    spec: Mapping[str, Any],
    generator: Any,
    *,
    device: str = "cpu",
) -> Any:
    import torch

    dtype_name = spec["dtype"]
    dtype = _torch_dtype(dtype_name)
    source_dtype = _torch_dtype(str(spec.get("source_dtype", dtype_name)))
    shape = tuple(spec["shape"])
    transforms = spec.get("transforms", [])
    if not isinstance(transforms, list):
        raise TypeError("random input transforms must be a list")

    prepared_random_values: dict[int, Any] = {}
    for index, transform in enumerate(transforms):
        if not isinstance(transform, Mapping) or not isinstance(
            transform.get("op"), str
        ):
            raise TypeError("random input transform must contain a string op")
        if transform["op"] == "multiply_random" and transform.get("before", False):
            nested = transform.get("input")
            if not isinstance(nested, Mapping):
                raise TypeError("multiply_random transform requires an input recipe")
            prepared_random_values[index] = _random_tensor(
                nested, generator, device=device
            )

    distribution = spec.get("distribution", "normal")
    if distribution == "normal":
        if source_dtype.is_floating_point or source_dtype.is_complex:
            value = torch.randn(
                shape,
                dtype=source_dtype,
                device=device,
                generator=generator,
            )
        elif source_dtype is torch.bool:
            value = torch.randint(
                0,
                2,
                shape,
                dtype=torch.bool,
                device=device,
                generator=generator,
            )
        else:
            ranges = {
                torch.int8: (-128, 128),
                torch.int16: (-1024, 1024),
                torch.int32: (-1024, 1024),
                torch.int64: (-1024, 1024),
                torch.uint8: (0, 256),
            }
            if source_dtype not in ranges:
                raise ValueError(f"unsupported random dtype: {dtype_name}")
            low, high = ranges[source_dtype]
            value = torch.randint(
                low,
                high,
                shape,
                dtype=source_dtype,
                device=device,
                generator=generator,
            )
    elif distribution == "uniform":
        if not source_dtype.is_floating_point:
            raise ValueError("uniform distribution requires a floating source dtype")
        low = float(spec.get("low", 0.0))
        high = float(spec.get("high", 1.0))
        value = torch.rand(
            shape,
            dtype=source_dtype,
            device=device,
            generator=generator,
        )
        if low != 0.0 or high != 1.0:
            value = value * (high - low) + low
    elif distribution == "integer":
        value = torch.randint(
            int(spec["low"]),
            int(spec["high"]),
            shape,
            dtype=source_dtype,
            device=device,
            generator=generator,
        )
    else:
        raise ValueError(f"unsupported random distribution: {distribution}")

    for index, transform in enumerate(transforms):
        operation = transform["op"]
        if operation == "transpose":
            value = value.T
        elif operation == "symmetrize_sum":
            value = value + value.T
        elif operation == "triu":
            value = torch.triu(value, diagonal=int(transform.get("diagonal", 0)))
        elif operation == "tril":
            value = torch.tril(value, diagonal=int(transform.get("diagonal", 0)))
        elif operation == "softmax":
            value = value.softmax(dim=int(transform["dim"]))
        elif operation == "cast":
            value = value.to(dtype=_torch_dtype(str(transform["dtype"])))
        elif operation == "multiply":
            value = value * transform["value"]
        elif operation == "divide":
            value = value / transform["value"]
        elif operation == "rdivide":
            value = transform["value"] / value
        elif operation == "add":
            value = value + transform["value"]
        elif operation == "subtract":
            value = value - transform["value"]
        elif operation == "rsubtract":
            value = transform["value"] - value
        elif operation == "multiply_random":
            operand = prepared_random_values.get(index)
            if operand is None:
                nested = transform.get("input")
                if not isinstance(nested, Mapping):
                    raise TypeError(
                        "multiply_random transform requires an input recipe"
                    )
                operand = _random_tensor(nested, generator, device=device)
            value = value * operand
        else:
            raise ValueError(f"unsupported random transform: {operation}")
    return value.to(dtype=dtype)


def _recipe(value: Any) -> tuple[str, Mapping[str, Any]] | None:
    if not isinstance(value, Mapping):
        return None
    kind = value.get("type")
    if kind not in {"random", "custom", "scalar", "literal", "safetensor"}:
        return None
    return str(kind), value


def _load_safetensors(path: str) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise RuntimeError(
            "safetensors is required for file-backed workloads"
        ) from exc

    values: dict[str, Any] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            values[key] = handle.get_tensor(key)
    return values


def make_cpu_bases(workload: Workload) -> dict[str, Any]:
    """Materialize ordinary V5-style recipes once on CPU.

    Arbitrary JSON entries remain generator context.  A fixed ``gen_inputs``
    hook may return the actual parameter mapping for dependent or framework
    inputs such as paged-cache indices.

    A hook may return ``None`` for workloads fully represented by ordinary
    recipes; otherwise it returns only the parameter overrides it generated.
    """

    import torch

    generator = torch.Generator(device="cpu").manual_seed(workload.seed)
    values: dict[str, Any] = {}
    for name, raw in workload.inputs.items():
        recipe = _recipe(raw)
        if recipe is None:
            values[name] = copy.deepcopy(raw)
            continue
        kind, spec = recipe
        if kind == "random":
            recipe_device = spec.get("device", "cpu")
            if recipe_device not in {"cpu", "target"}:
                raise ValueError("random input device must be cpu or target")
            if recipe_device == "target":
                continue
            shape = spec.get("shape")
            dtype = spec.get("dtype")
            if not isinstance(shape, list) or not isinstance(dtype, str):
                raise ValueError("random input requires list shape and string dtype")
            values[name] = _random_tensor(spec, generator)
        elif kind in {"scalar", "literal"}:
            if "value" not in spec:
                raise ValueError(f"{kind} input requires value")
            values[name] = copy.deepcopy(spec["value"])
    file_inputs = [
        name
        for name, raw in workload.inputs.items()
        if (recipe := _recipe(raw)) is not None and recipe[0] == "safetensor"
    ]
    if file_inputs and workload.input_path is None:
        raise ValueError("safetensor inputs require workload.input_path")
    if workload.input_path is not None:
        loaded = _load_safetensors(workload.input_path)
        missing = set(file_inputs) - set(loaded)
        if missing:
            raise ValueError(
                "input safetensors file is missing parameter keys: "
                f"{sorted(missing)}"
            )
        # File tensors override ordinary recipes with the same parameter name,
        # matching the v6.1/v6.2 workload contract.
        values.update(loaded)
    return values


def workload_context(workload: Workload) -> dict[str, Any]:
    return copy.deepcopy(workload.context())


def _move_tensors(
    value: Any,
    device: str,
    memo: dict[int, Any] | None = None,
) -> Any:
    import torch

    memo = {} if memo is None else memo
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if isinstance(value, torch.Tensor):
        result = value.to(device=device)
        memo[value_id] = result
        return result
    if isinstance(value, list):
        result: list[Any] = []
        memo[value_id] = result
        result.extend(_move_tensors(item, device, memo) for item in value)
        return result
    if isinstance(value, tuple):
        result = tuple(_move_tensors(item, device, memo) for item in value)
        memo[value_id] = result
        return result
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        memo[value_id] = result
        result.update(
            (key, _move_tensors(item, device, memo))
            for key, item in value.items()
        )
        return result
    return value


def materialize_values(
    definition: Definition,
    operator: Any,
    workload: Workload,
    cpu_bases: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    import torch

    parameter_names = {parameter.name for parameter in definition.parameters}
    values = {
        name: clone(value, device="cpu")
        for name, value in cpu_bases.items()
        if name in parameter_names
    }
    for name, raw in workload.inputs.items():
        recipe = _recipe(raw)
        if name not in parameter_names or recipe is None:
            continue
        kind, spec = recipe
        if kind != "random" or spec.get("device", "cpu") != "target":
            continue
        if not isinstance(spec.get("shape"), list) or not isinstance(
            spec.get("dtype"), str
        ):
            raise ValueError("random input requires list shape and string dtype")
        # _seed_globals() resets the assigned device's default generator before
        # each reference/candidate materialization.  Using that generator here
        # reproduces source builders that call torch.randn(..., device=device)
        # and avoids an observable CPU staging allocation.
        values[name] = _random_tensor(spec, None, device=device)
    if operator.gen_inputs is not None:
        generated = operator.gen_inputs(workload_context(workload), torch.device(device))
        if generated is not None and not isinstance(generated, Mapping):
            raise TypeError("reference gen_inputs must return a mapping")
        if generated is not None:
            unknown = set(generated) - parameter_names
            if unknown:
                raise ValueError(
                    f"gen_inputs returned unknown parameters: {sorted(unknown)}"
                )
            values.update(generated)

    missing = {
        parameter.name
        for parameter in definition.parameters
        if parameter.required and parameter.name not in values
    }
    if missing:
        raise ValueError(f"inputs did not materialize required parameters: {sorted(missing)}")
    return _move_tensors(values, device)


def make_call(
    definition: Definition,
    operator: Any,
    workload: Workload,
    cpu_bases: Mapping[str, Any],
    device: str,
) -> tuple[Call, inspect.BoundArguments]:
    values = materialize_values(definition, operator, workload, cpu_bases, device)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for parameter in definition.parameters:
        if parameter.name not in values:
            continue
        if parameter.kind == ParameterKind.VAR_POSITIONAL:
            items = values[parameter.name]
            if not isinstance(items, (list, tuple)):
                raise TypeError(
                    f"var_positional input {parameter.name!r} must be a list or tuple"
                )
            args.extend(items)
        elif parameter.kind == ParameterKind.POSITIONAL_ONLY:
            args.append(values[parameter.name])
        else:
            kwargs[parameter.name] = values[parameter.name]
    bound = operator.signature.bind(*args, **kwargs)
    return Call(tuple(args), kwargs), bound


__all__ = [
    "make_call",
    "make_cpu_bases",
    "materialize_values",
    "workload_context",
]
