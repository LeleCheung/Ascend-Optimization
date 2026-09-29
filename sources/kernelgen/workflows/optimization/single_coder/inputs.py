"""Build optimizer arguments from workflow configuration and prepared context."""

from typing import Any

from pydantic import BaseModel

from kernelgen.data.timeout_policy import TimeoutPolicy
from kernelgen.workflows.optimization.single_coder.workflow import SingleCoderOptimizationInput


def build_optimizer_input(config: BaseModel, **prepared: Any) -> dict[str, Any]:
    """Forward optimizer fields only; callers own Catalog, DPS and seed policy.

    Fields absent from the source model retain the optimizer's defaults. In
    particular, a workflow's ``timeout`` is not an Eval timeout. Validation
    remains at the existing SingleCoderOptimizationWorkflow input boundary.
    """
    result = config.model_dump(
        include=set(SingleCoderOptimizationInput.model_fields), mode="python",
    )
    if "eval_timeout_seconds" in result:
        result["eval_transport_timeout_seconds"] = TimeoutPolicy(
            result["eval_timeout_seconds"],
        ).eval_transport_timeout_seconds
    result.update(prepared)
    return result
