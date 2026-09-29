"""Representative real-LLM/MCP E2E for the FlagGems v4 extractor.

Run in the 910B-6 container, where the MCP process can load FlashInfer-Bench and
reach the isolated eval server directly. API credentials are read into process
environment from a Claude settings file and are never copied to env.sh.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


KERNELGEN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = KERNELGEN_ROOT.parent
sys.path.insert(0, str(PACKAGE_PARENT))

from kernelgen.framework import copy_claude_directory, materialize_mcp_configuration
from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.workflows.flaggems_extract import FlagGemsExtractWorkflow


SETTINGS = Path(
    os.environ.get(
        "CLAUDE_SETTINGS",
        "/data/xuyao/flashinfer-bench-v4/.claude/settings.xuyao.json",
    )
)
FLAGGEMS_REPO = Path(
    os.environ.get("FLAGGEMS_REPO", "/data/xuyao/FlagGems-master")
)
FLASHINFER_BENCH = Path(
    os.environ.get("FLASHINFER_BENCH", "/data/xuyao/flashinfer-bench-v4")
)
SERVER = os.environ.get("FIB_EVAL_SERVER", "http://127.0.0.1:18082")
TARGET = os.environ.get("TARGET_HARDWARE", "Ascend910B")
OPERATORS = [
    item.strip()
    for item in os.environ.get("OPERATORS", "gelu,uniform_").split(",")
    if item.strip()
]
RUN_ROOT = Path(
    os.environ.get(
        "WT",
        str(KERNELGEN_ROOT / "runs" / f"flaggems_extract_v4_{int(time.time())}"),
    )
)


def _load_model_environment() -> str:
    settings = json.loads(SETTINGS.read_text())
    values = settings.get("env") or {}
    for name, value in values.items():
        os.environ[str(name)] = str(value)
    model = str(
        os.environ.get("MODEL")
        or settings.get("model")
        or values.get("ANTHROPIC_MODEL")
        or "claude-opus-4-8[1m]"
    )
    if not os.environ.get("ANTHROPIC_BASE_URL") or not os.environ.get(
        "ANTHROPIC_AUTH_TOKEN"
    ):
        raise RuntimeError(f"Claude API configuration is incomplete in {SETTINGS}")
    return model


MODEL = _load_model_environment()
pythonpath = [
    str(FLASHINFER_BENCH),
    str(PACKAGE_PARENT),
    *(entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry),
]
os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
os.environ["FIB_EVAL_SERVER"] = SERVER
os.environ["FIB_TRACE_ROOT"] = "/data/xuyao"
os.environ["FIB_TRACE_SET_KEY"] = ""
os.environ["FIB_TARGET_HW"] = TARGET

RUN_ROOT.mkdir(parents=True, exist_ok=False)
summary = []

print(f"FlagGems v4 extractor E2E: operators={OPERATORS}, model={MODEL}")
print(f"Server: {SERVER}")
print(f"Run root: {RUN_ROOT}")

for operator in OPERATORS:
    workspace = RUN_ROOT / operator
    workspace.mkdir()
    copy_claude_directory(
        KERNELGEN_ROOT / ".claude",
        workspace / ".claude",
    )
    materialize_mcp_configuration(
        KERNELGEN_ROOT / ".kernelgen" / "mcp.json",
        workspace,
    )
    trace_root = workspace / "output_trace"
    verify_root = workspace / "verify_trace"

    def make_runtime(path: str) -> ClaudeRuntime:
        return ClaudeRuntime(
            workspace=path,
            model=MODEL,
            base_url=os.environ["ANTHROPIC_BASE_URL"],
            auth_token=os.environ["ANTHROPIC_AUTH_TOKEN"],
            timeout=2400,
            idle_timeout=900,
        )

    try:
        result = FlagGemsExtractWorkflow(
            cwd=str(workspace),
            runtime_factory=make_runtime,
        ).run(
            {
                "operator": operator,
                "flaggems_repo": str(FLAGGEMS_REPO),
                "trace_root": str(trace_root),
                "verify_trace_root": str(verify_root),
                "verify_target_hw": TARGET,
            }
        )
        passed = result.num_definitions > 0
        record = {
            "operator": operator,
            "passed": passed,
            "definitions": result.definitions,
            "trace_root": result.trace_root,
        }
    except Exception as error:  # keep representative batch evidence together
        record = {
            "operator": operator,
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }
    summary.append(record)
    print(json.dumps(record, ensure_ascii=False))

summary_path = RUN_ROOT / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
all_passed = all(item["passed"] for item in summary)
print(f"E2E RESULT: {'PASSED' if all_passed else 'FAILED'}")
print(f"Summary: {summary_path}")
raise SystemExit(0 if all_passed else 1)
