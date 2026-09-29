#!/usr/bin/env python3
"""Extract FlagGems addmm_ into the V6.2 per-operator native layout."""

from __future__ import annotations

import argparse
import json
import sys

from kernelgen.workflows.flaggems_v62_extract import (
    DEFAULT_CATALOG_ROOT,
    FlagGemsV62ExtractWorkflow,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", default="addmm_", choices=["addmm_"])
    parser.add_argument("--flaggems-repo", default="third_party/FlagGems")
    parser.add_argument("--catalog-root", default=DEFAULT_CATALOG_ROOT)
    parser.add_argument(
        "--case-list-path",
        help=(
            "Use an existing flaggems.benchmark-case-list/v2 JSON report instead "
            "of invoking pytest locally"
        ),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()
    result = FlagGemsV62ExtractWorkflow().run(vars(args))
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
