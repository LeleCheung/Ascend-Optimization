"""Workflow wrapper for the deterministic FlagGems V6.2 addmm_ extractor."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

from kernelgen.agents.extractor.flaggems.v62 import extract_flaggems_addmm_v62
from kernelgen.framework.workflow import Workflow


DEFAULT_CATALOG_ROOT = str(
    Path(__file__).resolve().parents[2]
    / "kernelgen_server"
    / "data"
    / "flaggems-native"
)


class FlagGemsV62ExtractInput(BaseModel):
    operator: str = Field(default="addmm_", pattern=r"^addmm_$")
    flaggems_repo: str = "third_party/FlagGems"
    catalog_root: str = DEFAULT_CATALOG_ROOT
    case_list_path: str | None = None
    python_executable: str = sys.executable


class FlagGemsV62ExtractOutput(BaseModel):
    operator: str
    catalog_root: str
    operator_root: str
    num_correctness_workloads: int
    num_timing_workloads: int


class FlagGemsV62ExtractWorkflow(Workflow):
    """Create one self-contained per-operator addmm_ package."""

    name = "flaggems_v62_extract"
    InputModel = FlagGemsV62ExtractInput
    OutputModel = FlagGemsV62ExtractOutput

    def _execute(self, inp: FlagGemsV62ExtractInput) -> Dict[str, Any]:
        result = extract_flaggems_addmm_v62(
            inp.flaggems_repo,
            inp.catalog_root,
            case_list_path=inp.case_list_path,
            python_executable=inp.python_executable,
        )
        return {
            "operator": inp.operator,
            "catalog_root": str(result.catalog_root),
            "operator_root": str(result.operator_root),
            "num_correctness_workloads": result.num_correctness_workloads,
            "num_timing_workloads": result.num_timing_workloads,
        }


__all__ = [
    "DEFAULT_CATALOG_ROOT",
    "FlagGemsV62ExtractInput",
    "FlagGemsV62ExtractOutput",
    "FlagGemsV62ExtractWorkflow",
]
