#!/usr/bin/env python3
"""Extract one KGS v6.2 native package directly from pinned FlagGems source."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kernelgen.agents.extractor.flaggems.v62_agent import (
    FlagGemsV62ExtractorAgent,
    FlagGemsV62ExtractorInput,
    build_flaggems_v62_accuracy_coverage,
    persist_flaggems_v62_extraction,
)
from kernelgen.framework.agent_roles import materialize_agent_role
from kernelgen.framework.runtime.claude import ClaudeRuntime


KERNELGEN_ROOT = Path(__file__).resolve().parents[2]


def _runtime(workspace: Path, model: str, timeout: int) -> ClaudeRuntime:
    source = (
        KERNELGEN_ROOT
        / ".kernelgen"
        / "agents"
        / "kernel-flaggems-v62-extractor.md"
    )
    materialize_agent_role(source, workspace)
    return ClaudeRuntime(
        workspace=workspace,
        model=model,
        allowed_tools="Read",
        permission_mode="acceptEdits",
        timeout=timeout,
        idle_timeout=max(120, timeout // 2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--flaggems-repo", required=True)
    parser.add_argument("--case-list-path", required=True)
    parser.add_argument("--catalog-root", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("MODEL", "inherit"))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")

    args.workspace.mkdir(parents=True, exist_ok=True)
    catalog_root = Path(args.catalog_root).expanduser().resolve()
    adapter_catalog_root = catalog_root.parent / "flaggems-adapter-definitions"
    if not (adapter_catalog_root / "manifest.json").is_file():
        adapter_catalog_root = None
    inp = FlagGemsV62ExtractorInput(
        operator=args.operator,
        flaggems_repo=args.flaggems_repo,
        case_list_path=args.case_list_path,
    )
    output = FlagGemsV62ExtractorAgent().run(
        inp.model_dump(mode="python"),
        _runtime(args.workspace, args.model, args.timeout),
    )
    result = persist_flaggems_v62_extraction(
        inp,
        output,
        catalog_root,
        adapter_catalog_root=adapter_catalog_root,
    )
    coverage_path = args.workspace / "accuracy_coverage.json"
    coverage_path.write_text(
        json.dumps(
            build_flaggems_v62_accuracy_coverage(inp, output),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "operator": args.operator,
                "operator_root": str(result.operator_root),
                "adapter_definition": (
                    str(result.adapter_definition_path)
                    if result.adapter_definition_path is not None
                    else None
                ),
                "correctness_workloads": result.num_correctness_workloads,
                "timing_workloads": result.num_timing_workloads,
                "accuracy_coverage": str(coverage_path.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
