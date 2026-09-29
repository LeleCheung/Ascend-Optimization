"""DistillerAgent (ADR-3 #10): distill one agent's optimization run into candidate KB.

Reads the agent's optimization trajectory (initial kernel + per-round code diff +
strategy/root_cause) from the prompt (pre-extracted by the workflow from ledger
rounds) and produces candidate experience/detailed text.

Does NOT read files from workspace (no Glob/Read needed — all info is in the prompt).
Does NOT write KB itself; the epoch reducer handles canonical persistence.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from kernelgen.framework.base import BaseAgent
from kernelgen.data.implementation import (
    ImplementationLanguage,
    render_implementation_profile,
)


class DistillInput(BaseModel):
    """What the distiller needs: the optimization history + context."""
    definition_name: str
    op_type: str
    target_hardware: str
    implementation_language: ImplementationLanguage = ImplementationLanguage.TRITON
    definition: Dict[str, Any] = Field(default_factory=dict)
    # The optimization rounds (from ledger.history.rounds, serialized)
    rounds: List[Dict[str, Any]] = Field(default_factory=list)
    best_geo_mean: float = 0.0
    best_round: int = 0


class DistillOutput(BaseModel):
    """Candidate experience + detailed text to be merged into KB."""
    candidate_experience: str = Field(default="", description="distilled experience (lessons, patterns)")
    candidate_detailed: str = Field(default="", description="detailed analysis (per-round reasoning chain)")
    skip_reason: str = Field(default="", description="if empty = has content; if set = why skipping distillation")


class DistillerAgent(BaseAgent):
    name = "distiller"
    InputModel = DistillInput
    OutputModel = DistillOutput

    def preprocess(self, inp: DistillInput, runtime) -> str:
        from kernelgen.framework.contract import render_contract
        from kernelgen.data.trajectory import build_trajectory

        role = self._role_for(runtime)

        if not inp.rounds:
            return ('No optimization rounds to distill. Output: '
                    '{"skip_reason": "no rounds", "candidate_experience": "", "candidate_detailed": ""}')

        # Build trajectory (initial kernel + per-round diff/REWRITE)
        trajectory = build_trajectory(inp.rounds)

        # Context summary
        context = (
            f"Definition: {inp.definition_name} (op_type={inp.op_type}, hardware={inp.target_hardware})\n"
            f"Best: geo_mean={inp.best_geo_mean:.3f}x at round {inp.best_round}\n"
            f"Total rounds: {len(inp.rounds)}\n"
            "Canonical definition (do not infer conflicting shapes or constants):\n"
            f"{json.dumps(inp.definition, indent=2, ensure_ascii=False, default=str)[:12000]}"
        )

        contract = render_contract(self.OutputModel)
        prompt = "\n\n".join(filter(None, [
            role,
            f"<context>\n{context}\n</context>",
            render_implementation_profile(inp.implementation_language),
            f"<trajectory>\n{trajectory}\n</trajectory>",
            "<analysis_requirement>\nTreat every round as plan → solution → evaluation → profile → conclusion → next verdict. Compare the frozen expected effect with Python-owned aggregate and per-workload measurements, preserve latency as well as speedup, and identify validated, partially validated, falsified, or unevaluable hypotheses.\n</analysis_requirement>",
            f"--- OUTPUT CONTRACT ---\n{contract}",
        ]))
        return prompt
