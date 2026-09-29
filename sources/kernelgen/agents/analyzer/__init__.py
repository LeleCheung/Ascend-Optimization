"""AnalyzerAgent (ADR-3 #6): kernel specification → structured optimization plan.

Cold-start only: given a new kernel definition (no prior solution), produces a
structured plan (grid/loop/pitfalls) for the CoderAgent. Breakthrough analysis
(next-epoch directions based on existing best) is handled by EpochSummaryAgent.

Role: .kernelgen/agents/kernel-analyzer.md (materialized for providers, inline fallback
for runtimes without named-agent support).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import AliasChoices, BaseModel, Field
from kernelgen.agents._definition_prompt import (
    render_definition_block,
    render_evaluation_contract,
)
from kernelgen.agents._workload_prompt import sample_workloads_for_prompt

from kernelgen.framework.base import BaseAgent
from kernelgen.framework.models import DefinitionModel, EvaluationContractModel
from kernelgen.data.implementation import (
    ImplementationLanguage,
    render_implementation_profile,
    render_target_compatibility_rules,
)


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------

class AnalyzerInput(BaseModel):
    definition: DefinitionModel             # nested; a plain dict auto-converts
    target_hardware: str                    # "Ascend910B" / "H100" / ...
    implementation_language: ImplementationLanguage = ImplementationLanguage.TRITON
    workloads: List[Dict[str, Any]] = Field(default_factory=list)
    server_context: Dict[str, Any] = Field(default_factory=dict)

    destination_passing_style: bool = True
    evaluation_contract: EvaluationContractModel = Field(
        default_factory=EvaluationContractModel
    )


class AnalyzerOutput(BaseModel):
    core_math: str = Field(default="", description="core mathematical formula")
    program_grid: str = Field(
        default="",
        description="proposed device-program grid dimensions",
        validation_alias=AliasChoices("program_grid", "triton_grid"),
    )
    inner_loop: str = Field(default="", description="inner loop structure")
    softmax_strategy: str = Field(default="N/A", description="online vs standard softmax, or N/A")
    data_layout: str = Field(default="", description="tensor layout notes")
    key_pitfalls: List[str] = Field(default_factory=list, description="Triton implementation pitfalls")
    workload_dispatch_strategy: str = Field(default="", description="kernel dispatch strategy across workloads")
    head_mapping: str = Field(default="N/A")
    masking: str = Field(default="N/A")
    lse_formula: str = Field(default="N/A")
    memory_traffic_analysis: Dict[str, Any] = Field(default_factory=dict)

    numerical_strategy: str = Field(
        default="",
        description="recommended compute and accumulation dtypes under the evaluator contract",
    )
    reduced_precision_candidates: List[str] = Field(
        default_factory=list,
        description="specific reduced-precision experiments worth evaluating",
    )
    @property
    def triton_grid(self) -> str:
        """Backward-compatible accessor for pre-migration analysis consumers."""
        return self.program_grid


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AnalyzerAgent(BaseAgent):
    name = "analyzer"
    InputModel = AnalyzerInput
    OutputModel = AnalyzerOutput

    def preprocess(self, inp: AnalyzerInput, runtime) -> str:
        role = self._role_for(runtime)

        task_tag = (
            "Analyze a GPU kernel specification. Do NOT write implementation code. "
            "Only output analysis."
        )

        definition_block = render_definition_block(
            inp.definition,
            destination_passing_style=inp.destination_passing_style,
        )

        from kernelgen.framework.contract import render_contract
        contract = render_contract(self.OutputModel)
        shown = sample_workloads_for_prompt(inp.workloads)
        prompt = "\n\n".join(filter(None, [
            role,
            f"<task>\n{task_tag}\n</task>",
            f"<hardware>\nTarget GPU: {inp.target_hardware}\n</hardware>",
            "<server_context>\n"
            "Facts from the evaluation Server /status, not the Agent host. "
            "Missing metadata is unknown. Do not invent core counts, peak "
            "throughput, bandwidth or compiler capabilities. Portability rules "
            "do not change the identity of the measured device. Never import or "
            "probe the Agent host's Torch, Triton or device environment to fill "
            "gaps. Leave unresolved target queries for the Coder's KGS Debug "
            "Jobs, and label estimates separately from reported facts.\n"
            + json.dumps(inp.server_context, ensure_ascii=False)
            + "\n</server_context>",
            render_target_compatibility_rules(
                inp.implementation_language,
                inp.target_hardware,
            ),
            render_implementation_profile(inp.implementation_language),
            render_evaluation_contract(inp.evaluation_contract),
            definition_block,
            "<workloads>\n"
            "Authoritative evaluation inputs, including correctness/timing phase, "
            "tensor shapes/dtypes and scalar/literal values. Analyze these cases "
            "rather than inventing representative shapes. Empty means unavailable, "
            "not unrestricted. Preserve the complete evaluation coverage. "
            f"Showing {len(shown)} deterministically sampled cases of {len(inp.workloads)}; "
            "unshown cases remain required.\n"
            + json.dumps(shown, ensure_ascii=False)
            + "\n</workloads>",
            f"--- OUTPUT CONTRACT ---\n{contract}",
        ]))
        return prompt
