"""Shared rendering for kernel definition prompt blocks."""

from __future__ import annotations

import textwrap

from kernelgen.framework.models import DefinitionModel, EvaluationContractModel


def render_evaluation_contract(
    contract: EvaluationContractModel,
) -> str:
    """Render the authoritative numerical gate and precision search policy."""
    if contract.atol is None:
        tolerance = (
            "Tolerance thresholds: not declared. Do not infer them from input or "
            "output dtypes; measure every numerical experiment."
        )
    else:
        mode = contract.tolerance_mode or "unspecified"
        tolerance = (
            f"Tolerance mode: {mode}\n"
            f"Elementwise acceptance: |candidate-reference| <= "
            f"{contract.atol:g} + {contract.rtol:g} * |reference|\n"
            f"Required matched element ratio: "
            f"{contract.required_matched_ratio:g}"
        )

    precision_policy = (
        "Reduced-precision internal operands, intermediates, or accumulation "
        "(for example FP16, BF16, or hardware reduced-precision FP32) MAY be "
        "considered when their end-to-end error is likely to satisfy this "
        "contract."
        if contract.consider_reduced_precision
        else
        "Do not deliberately reduce internal precision below the reference "
        "numerical path."
    )

    checks = [
        (
            "Declared output dtype must match exactly."
            if contract.check_output_dtype
            else "Output dtype is not checked by this contract."
        ),
        (
            "NaN and Inf outputs are rejected."
            if contract.reject_non_finite
            else "Non-finite output handling is evaluator-specific."
        ),
    ]
    return (
        "<evaluation_contract>\n"
        + tolerance
        + "\n"
        + "\n".join(f"- {item}" for item in checks)
        + "\n"
        + f"- {precision_policy}\n"
        + "- Preserve observable state mutations and the declared output dtype.\n"
        + "- Near-zero reference values are governed mainly by atol; do not "
        "treat every small value as zero.\n"
        + "- Include per-invocation cast/conversion costs. Do not assume mutable "
        "inputs can be cached across calls.\n"
        + "- Precision estimates are screening evidence only. Preflight does not "
        "prove numerical correctness; eval_round is authoritative.\n"
        + "</evaluation_contract>"
    )


def render_definition_block(
    definition: DefinitionModel,
    *,
    destination_passing_style: bool,
) -> str:
    axes = "\n".join(
        f"  {name}: {value.get('type', '?')}"
        + (f" = {value['value']}" if "value" in value else "")
        for name, value in definition.axes.items()
    )
    inputs = "\n".join(
        (
            f"  {name}: {value.get('shape', [])} ({value.get('dtype', '')})"
            if isinstance(value, dict)
            else f"  {name}: {value!r} ({type(value).__name__})"
        )
        for name, value in definition.inputs.items()
    )
    output_items = (
        definition.outputs.items()
        if isinstance(definition.outputs, dict)
        else ((name, {}) for name in definition.outputs)
    )
    outputs = "\n".join(
        f"  {name}: {value.get('shape', [])} ({value.get('dtype', '')})"
        for name, value in output_items
    )
    reference = textwrap.indent(definition.reference.strip(), "  ")
    if definition.run_signature:
        interface = f"""Exact run ABI:
  run{definition.run_signature}"""
    elif destination_passing_style:
        interface = f"""Evaluation Interface: destination-passing (DPS)
  run() MUST have exactly {definition.dps_param_count} positional parameters:
  {len(definition.inputs)} input(s) followed by {len(definition.outputs)} pre-allocated output buffer(s).
  Write results into the output buffer(s)."""
    else:
        interface = f"""Evaluation Interface: value-returning
  run() MUST have exactly {len(definition.inputs)} positional parameters:
  {len(definition.inputs)} input(s) and no output-buffer parameters.
  Return the output value, or a tuple for multiple outputs."""
    return f"""<definition>
Name: {definition.name}
Type: {definition.op_type}

Axes:
{axes}

Inputs:
{inputs}

Outputs:
{outputs}

{interface}

Reference Implementation:
{reference}
</definition>"""
