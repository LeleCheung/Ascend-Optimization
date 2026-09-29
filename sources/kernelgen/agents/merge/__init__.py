"""MergeAgent (ADR-3 #10): Judge+Merge KB content.

A lightweight agent that compares a NEW candidate (experience/detailed/skill) against
EXISTING KB content and decides: KEEP existing / DISCARD candidate / MERGE both.
If MERGE, produces the merged text.

Used by the epoch knowledge reducer and other synthesis workflows that need bounded
semantic merging. The agent proposes merged text; Python remains responsible for
candidate ordering, validation, and persistence.

**Model selection**: by default uses the runtime from bind() (same as other agents).
If env vars MERGE_MODEL / MERGE_BASE_URL / MERGE_AUTH_TOKEN are set, constructs its
own cheaper runtime — a cost optimization (merge is short judgment, doesn't need the
expensive model). Framework doesn't know or care about this; it's MergeAgent's
internal decision.
"""

from __future__ import annotations

import os
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kernelgen.framework.base import BaseAgent


class MergeInput(BaseModel):
    """What to merge: candidate text vs existing text, with context."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["judge_merge", "n_way_merge"] = "judge_merge"
    # judge_merge mode: compare candidate vs existing
    candidate: str = ""
    existing: str = ""
    label: str = ""                     # e.g. "experience" / "detailed" / skill name
    definition_name: str = ""
    op_type: str = ""
    target_hardware: str = ""
    # n_way_merge mode: merge N texts
    texts: List[str] = Field(default_factory=list)
    merge_instruction: str = ""         # what to produce from the N texts

    @model_validator(mode="after")
    def validate_mode_payload(self) -> "MergeInput":
        if self.mode == "judge_merge":
            if not self.candidate.strip():
                raise ValueError("judge_merge requires a non-empty candidate")
            if self.texts:
                raise ValueError("judge_merge does not accept texts")
        else:
            if not self.texts or not any(text.strip() for text in self.texts):
                raise ValueError("n_way_merge requires at least one non-empty text")
            if self.candidate.strip() or self.existing.strip():
                raise ValueError("n_way_merge does not accept candidate or existing")
        return self


class MergeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["KEEP", "DISCARD", "MERGE"] = Field(
        description="KEEP existing, DISCARD candidate, or MERGE supplied content"
    )
    merged_text: str = Field(
        default="",
        description="complete merged document for MERGE; empty for KEEP or DISCARD",
    )

    @model_validator(mode="after")
    def validate_verdict_payload(self) -> "MergeOutput":
        if self.verdict == "MERGE" and not self.merged_text.strip():
            raise ValueError("MERGE requires non-empty merged_text")
        if self.verdict != "MERGE" and self.merged_text.strip():
            raise ValueError("KEEP and DISCARD require empty merged_text")
        return self


class MergeAgent(BaseAgent):
    name = "merge"
    InputModel = MergeInput
    OutputModel = MergeOutput

    def _execute(self, inp: MergeInput) -> MergeOutput:
        """Override to optionally use a cheaper runtime from env."""
        runtime = self._get_merge_runtime()
        base_prompt = self.preprocess(inp, runtime)
        prompt = base_prompt

        from kernelgen.framework.contract import extract_json, append_repair
        from pydantic import ValidationError
        from kernelgen.framework.base import _MAX_REPAIRS, AgentContractError

        last_err = None
        total_attempts = 1 + _MAX_REPAIRS
        for attempt in range(total_attempts):
            raw = self._invoke_runtime(runtime, prompt)
            try:
                result = self.postprocess(raw, runtime)
                if inp.mode == "n_way_merge" and result.verdict != "MERGE":
                    raise ValueError("n_way_merge requires verdict='MERGE'")
                return result
            except (ValidationError, ValueError) as e:
                last_err = e
                if attempt < _MAX_REPAIRS:
                    prompt = append_repair(base_prompt, e, raw)
        raise AgentContractError(
            "MergeAgent: no valid output after "
            f"{total_attempts} attempts: {last_err}"
        )

    def _get_merge_runtime(self):
        """Use env-configured cheap runtime if available; else default."""
        merge_model = os.environ.get("MERGE_MODEL")
        if not merge_model:
            return self._runtime  # use default (from bind or run)
        # Build a dedicated cheap runtime from env
        from kernelgen.framework.runtime.claude import ClaudeRuntime
        return ClaudeRuntime(
            workspace=self._runtime.workspace if self._runtime else None,
            model=merge_model,
            base_url=os.environ.get("MERGE_BASE_URL"),
            auth_token=os.environ.get("MERGE_AUTH_TOKEN"),
            timeout=300,        # merge is fast
            idle_timeout=120,
        )

    def preprocess(self, inp: MergeInput, runtime) -> str:
        from kernelgen.framework.contract import render_contract

        role = self._role_for(runtime)
        context = (
            "<context>\n"
            f"Mode: {inp.mode}\n"
            f"Definition: {inp.definition_name or 'N/A'}\n"
            f"Operator type: {inp.op_type or 'N/A'}\n"
            f"Hardware: {inp.target_hardware or 'N/A'}\n"
            f"Label: {inp.label or 'N/A'}\n"
            "</context>"
        )

        if inp.mode == "n_way_merge":
            task = (
                "<task>\n"
                "Merge all supplied documents into one coherent document while preserving "
                "compatible evidence, scope conditions, and unresolved contradictions.\n"
                "</task>"
            )
            documents = "\n".join(
                f'<document index="{index}">\n{text}\n</document>'
                for index, text in enumerate(inp.texts, start=1)
            )
            instruction = inp.merge_instruction.strip() or (
                "Deduplicate equivalent claims, retain necessary evidence qualifiers, and "
                "return verdict MERGE with the complete merged document."
            )
        else:
            task = (
                "<task>\n"
                "Compare the candidate with the existing document and decide whether to "
                "keep the existing document, discard the candidate, or merge their "
                "supported information.\n"
                "</task>"
            )
            documents = (
                "<existing>\n"
                f"{inp.existing}\n"
                "</existing>\n\n"
                "<candidate>\n"
                f"{inp.candidate}\n"
                "</candidate>"
            )
            instruction = (
                "Use KEEP when the candidate adds no reliable information. "
                "Use DISCARD when the candidate is incorrect, contradicted by stronger "
                "evidence, or outside scope. Use MERGE when the candidate adds useful "
                "supported information."
            )

        return "\n\n".join(
            filter(
                None,
                [
                    role,
                    context,
                    task,
                    f"<documents>\n{documents}\n</documents>",
                    f"<merge_instruction>\n{instruction}\n</merge_instruction>",
                    f"--- OUTPUT CONTRACT ---\n{render_contract(self.OutputModel)}",
                ],
            )
        )
