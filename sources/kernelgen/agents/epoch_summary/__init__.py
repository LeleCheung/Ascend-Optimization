"""EpochSummaryAgent (ADR-3 #10): cross-agent synthesis (per-epoch).

Reads N agents' results (best geo, strategy, architecture), compares them, and:
1. Produces next-epoch DIRECTIONS (breakthrough strategies to explore)
2. Writes a synthesis REPORT (human-readable summary of what happened this epoch)

Does NOT merge KB itself; the epoch reducer has already produced the KB snapshot.
This agent is the "strategic judgment" step — heavy LLM, needs the expensive model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from kernelgen.framework.base import BaseAgent
from kernelgen.data.implementation import (
    ImplementationLanguage,
    render_implementation_profile,
)


class EpochSummaryInput(BaseModel):
    """Cross-agent synthesis input."""
    definition_name: str
    op_type: str
    target_hardware: str
    implementation_language: ImplementationLanguage = ImplementationLanguage.TRITON
    n_agents: int
    # Per-agent summaries (from their CoderReports + ledger snapshots)
    agent_results: List[Dict[str, Any]] = Field(default_factory=list)
    # E.g. [{"agent_id": "task0", "status": "PASSED", "best_geo": 1.5,
    #         "strategy": "...", "architecture_tag": "...", "workspace": "..."}]
    fixed_best_geo: Optional[float] = None   # authoritative best (from pick_best)
    fixed_best_agent: str = ""               # which agent had the best (from pick_best)
    fixed_best_round: int = 0
    fixed_best_kernel: str = ""               # one code anchor; never repeated per agent


class EpochSummaryOutput(BaseModel):
    """Synthesis result: next directions plus a human-readable report."""
    next_directions: List[Dict[str, Any]] = Field(default_factory=list,
        description="breakthrough directions for next epoch [{direction, expected_gain, key_implementation_notes}]")
    synthesis_report: str = Field(default="", description="human-readable epoch summary")


class EpochSummaryAgent(BaseAgent):
    name = "epoch_summary"
    InputModel = EpochSummaryInput
    OutputModel = EpochSummaryOutput

    def preprocess(self, inp: EpochSummaryInput, runtime) -> str:
        from kernelgen.framework.contract import render_contract
        role = self._role_for(runtime)
        best_geo = f"{inp.fixed_best_geo:.3f}x" if inp.fixed_best_geo is not None else "none"

        # Build per-agent blocks with trajectory + new experience
        agent_blocks = []
        for r in inp.agent_results:
            block = (
                f"### Agent {r.get('agent_id','?')}\n"
                f"status={r.get('status','?')} | best_geo={r.get('best_geo',0):.3f}x | "
                f"strategy: {r.get('strategy','')[:150]}\n"
            )
            trajectory = r.get("trajectory", "")
            if trajectory:
                block += f"\n**Optimization trajectory:**\n{trajectory}\n"
            new_exp = r.get("new_experience", "")
            if new_exp:
                block += (
                    "\n**New experience (distilled this run):**\n"
                    f"{_bounded_text(new_exp, 3000)}\n"
                )
            agent_blocks.append(block)

        agents_section = "\n---\n".join(agent_blocks)
        best_kernel_block = ""
        if inp.fixed_best_kernel:
            best_kernel_block = (
                "<authoritative_best_kernel>\n"
                "This is the single code anchor that will seed the next epoch. "
                "Use it to avoid proposing work that is already implemented.\n"
                f"```python\n{inp.fixed_best_kernel}\n```\n"
                "</authoritative_best_kernel>"
            )

        prompt_parts = [
            role,
            f"<context>\nDefinition: {inp.definition_name} (op_type={inp.op_type}, hardware={inp.target_hardware})\n"
            f"Agents: {inp.n_agents}\n"
            f"Authoritative best: geo={best_geo} "
            f"(agent={inp.fixed_best_agent or 'none'}, "
            f"round={inp.fixed_best_round or 'none'})\n</context>",
            render_implementation_profile(inp.implementation_language),
            best_kernel_block,
            f"<agent_results>\n{agents_section}\n</agent_results>",
            "<comparison_requirement>\nCompare parallel agents by authoritative results and by calibration: what each frozen plan expected, what latency and speedup actually changed, and what its perf-gap conclusion explained. The per-agent trajectories are intentionally code-free; use the single authoritative kernel above only as the implementation anchor. Use repeated agreement or disagreement across agents to eliminate dead ends and diversify the next directions.\n</comparison_requirement>",
            f"--- OUTPUT CONTRACT ---\n{render_contract(self.OutputModel)}",
        ]
        return "\n\n".join(p for p in prompt_parts if p)


def _bounded_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    marker = "\n[distilled experience truncated to synthesis budget]"
    return text[: max_chars - len(marker)].rstrip() + marker
