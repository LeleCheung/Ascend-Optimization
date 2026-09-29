"""Real SimpleOpt E2E through kg run; requires an already prepared KGS/runtime."""

import os
from pathlib import Path

from kernelgen.cli.main import main
from kernelgen.data.constants import DEFAULT_CATALOG_NAME


def run():
    definition = os.environ.get("DEFINITION_NAME", "negative")
    workspace = Path(os.environ.get("WT", f"runs/e2e_simple_opt/{definition}")).resolve()
    arguments = [
        "run", "--mode", "simple_opt", "--foreground",
        "--definition", definition, "--workspace", str(workspace),
        "--catalog-name", os.environ.get("KERNELGEN_CATALOG_NAME", DEFAULT_CATALOG_NAME),
        "--eval-server", os.environ.get("FIB_EVAL_SERVER", "http://localhost:8000"),
        "--early-stop-rounds", "2", "--min-rounds", "2",
    ]
    if os.environ.get("TARGET_HARDWARE"):
        arguments.extend(["--target-hardware", os.environ["TARGET_HARDWARE"]])
    # CLI owns runtime credentials, workspace exclusivity and durable state.
    # Existing evidence must never be deleted to make a new run fit this path.
    code = main(arguments)
    optimizer = workspace / "stages/optimize/work"
    passed = code == 0 and all((optimizer / name).is_file() for name in (
        ".ledger.json", ".best_kernel.py", "optimize_definition_output.json",
    ))
    print(f"E2E RESULT: {'PASSED' if passed else 'FAILED'}; workspace: {workspace}")
    return 0 if passed else code or 1


if __name__ == "__main__":
    raise SystemExit(run())
