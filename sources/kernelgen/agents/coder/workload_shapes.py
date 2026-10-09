"""Normalize concrete tensor metadata for the coder prompt.

V4 workloads already carry shape and dtype on every random/custom tensor input;
those fields are copied directly into ``resolved_inputs`` without executing the
reference. Original selector-based traces remain supported through a best-effort
legacy fallback that invokes ``gen_inputs`` on CPU.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List


def _resolve_axes_for_ctx(definition: Dict[str, Any], workload_axes: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort ctx = resolved const/var axes + the workload's own axis values.

    Mirrors what the server passes to gen_inputs: const axes resolved to their
    value, var axes taken from the workload. We don't evaluate expr axes here
    (gen_inputs rarely needs them for the shape it returns); if it does and fails,
    we degrade gracefully.
    """
    ctx: Dict[str, Any] = {}
    for name, spec in (definition.get("axes") or {}).items():
        if isinstance(spec, dict) and spec.get("type") == "const" and "value" in spec:
            ctx[name] = spec["value"]
    ctx.update(workload_axes or {})
    return ctx


def enrich_workloads_with_shapes(
    definition: Dict[str, Any], workloads: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return copied workloads with normalized ``resolved_inputs`` metadata.

    V4 metadata is authoritative. Only random/custom inputs missing shape or dtype
    enter the legacy generator fallback.
    """
    enriched: List[Dict[str, Any]] = []
    needs_legacy = False
    for wl in workloads:
        item = copy.deepcopy(wl)
        target = item.get("workload", item)
        resolved: Dict[str, Any] = {}
        inputs = target.get("inputs") or {}
        if not isinstance(inputs, dict):
            enriched.append(item)
            continue
        for input_name, spec in inputs.items():
            # Lists and scalar literals are valid Definition inputs, but they do
            # not carry tensor metadata to normalize for the coder prompt.
            if not isinstance(spec, dict):
                continue
            if spec.get("type") not in {"random", "custom"}:
                continue
            if spec.get("shape") is not None and spec.get("dtype"):
                resolved[input_name] = {
                    "shape": copy.deepcopy(spec["shape"]),
                    "dtype": spec["dtype"],
                }
            else:
                needs_legacy = True
        if resolved:
            target["resolved_inputs"] = resolved
        enriched.append(item)

    if not needs_legacy:
        return enriched

    reference = definition.get("reference") or ""
    entry = definition.get("custom_inputs_entrypoint") or "gen_inputs"

    # Lazily import torch; if unavailable (host without torch), skip enrichment.
    try:
        import torch  # noqa: F401
    except Exception:
        return enriched

    # Exec the reference once to get the generator function.
    namespace: Dict[str, Any] = {}
    try:
        exec(compile(reference, "<reference>", "exec"), namespace)
    except Exception:
        return enriched
    gen = namespace.get(entry)
    if not callable(gen):
        return enriched

    cpu = torch.device("cpu")
    for wl, item in zip(workloads, enriched):
        w = wl.get("workload", wl)  # accept both wrapped-Trace and bare-workload shapes
        axes = (w or {}).get("axes", {})
        try:
            ctx = _resolve_axes_for_ctx(definition, axes)
            produced = gen(ctx, cpu)
            target = item.get("workload", item)
            resolved: Dict[str, Any] = target.setdefault("resolved_inputs", {})
            for iname, val in (produced or {}).items():
                if hasattr(val, "shape") and hasattr(val, "dtype"):
                    resolved.setdefault(iname, {
                        "shape": list(val.shape),
                        "dtype": str(val.dtype).replace("torch.", ""),
                    })
        except Exception:
            pass  # leave this workload unenriched
    return enriched
