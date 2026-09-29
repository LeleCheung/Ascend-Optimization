"""Prepare a frozen PR checkout, or collect cases and extract a Catalog."""

import argparse
import json
from pathlib import Path

from kernelgen.agents.extractor.flaggems.pr_source import changed_paths, prepare_pull_request
from kernelgen.examples.flaggems_v62_extract.run_direct_agent import _runtime
from kernelgen.workflows.catalog_extract import CatalogExtractWorkflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "extract"])
    parser.add_argument("--pr-url", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--operator")
    parser.add_argument("--case-list-path", type=Path)
    parser.add_argument("--model", default="inherit")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.action == "extract" and not args.operator:
        parser.error("extract requires --operator")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    root = args.workspace.expanduser().resolve()
    if args.action == "prepare":
        source = prepare_pull_request(args.pr_url, root / "source")
        print(json.dumps({**source.model_dump(), "head_checkout": str(root / "source/head"),
                          "changed_paths": changed_paths(root / "source", source)}, indent=2))
        return 0
    extracted = CatalogExtractWorkflow(cwd=str(root), runtime_factory=lambda path: _runtime(Path(path) / "model", args.model, args.timeout)).run({
        "operator": args.operator, "pr_url": args.pr_url,
        "case_list_path": str(args.case_list_path.expanduser().resolve()) if args.case_list_path else None})
    print(json.dumps({"operator": args.operator, "catalog_path": str(extracted.catalog_path),
                      "source_manifest": str(root / "source/source.json"),
                      "target_validation": "NOT_RUN"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
