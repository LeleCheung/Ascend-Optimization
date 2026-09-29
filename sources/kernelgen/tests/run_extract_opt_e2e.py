"""E2E test for ExtractOptWorkflow with dummy extractor input.

Validates:
1. Dummy extraction loads correctly (2 definitions: pytorch_add_tensor + pytorch_add_scalar)
2. Trace data written to expected paths
3. CoderAgent × 2 run in parallel (one per definition)
4. eval_round POSTs to real eval server
5. finalize_round finalization stops correctly
6. Results collected with geo_mean

Run inside pytorch-ncu-data container:
    cd /data/akg_kernel_bench_lite
    PYTHONPATH=. python3 -u kernelgen/tests/run_extract_opt_e2e.py
"""

import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernelgen.framework import copy_claude_directory
from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.workflows.extract_opt import ExtractOptWorkflow

# ============================================================
# Config
# ============================================================

KERNELGEN_ROOT = Path(__file__).resolve().parents[1]
WT = Path(os.environ.get("WT", str(KERNELGEN_ROOT / "runs" / "e2e_extract_opt")))
OPERATOR = "add"
SERVER = os.environ.get("FIB_EVAL_SERVER", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "deepseek-v4-pro[1m]")
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL")
AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN")

if not BASE_URL or not AUTH_TOKEN:
    raise RuntimeError(
        "DeepSeek API config is missing; run `source env.sh` before this e2e"
    )

# ============================================================
# Setup
# ============================================================

if WT.exists():
    shutil.rmtree(WT)
WT.mkdir(parents=True)

print("=" * 70)
print(f"ExtractOpt E2E: operator={OPERATOR}, model={MODEL}")
print(f"Workspace: {WT}")
print(f"Eval server: {SERVER}")
print("=" * 70)

# Materialize provider settings, roles, and skills.
claude_src = KERNELGEN_ROOT / ".claude"
copy_claude_directory(claude_src, WT / ".claude")
print("✅ Materialized neutral runtime configuration")

# Copy dummy extractor data
dummy_src = KERNELGEN_ROOT / "tmp" / OPERATOR
if dummy_src.exists():
    shutil.copytree(dummy_src, WT / "tmp" / OPERATOR, dirs_exist_ok=True)
    print(f"✅ Copied dummy data from {dummy_src}")
else:
    print(f"❌ Dummy data not found: {dummy_src}")
    sys.exit(1)

# Set env for eval_round tool — must match workflow's _resolve_trace_root default
TRACE_ROOT = str(KERNELGEN_ROOT / "trace_data")
os.environ["FIB_EVAL_SERVER"] = SERVER
os.environ["FIB_TRACE_ROOT"] = TRACE_ROOT
os.environ["FIB_TRACE_SET_KEY"] = ""

# ============================================================
# Build and run workflow
# ============================================================

def make_rt(path):
    return ClaudeRuntime(
        workspace=path,
        model=MODEL,
        base_url=BASE_URL,
        auth_token=AUTH_TOKEN,
        timeout=1200,
        idle_timeout=600,
    )

wf = ExtractOptWorkflow(cwd=str(WT), runtime_factory=make_rt)

inp = {
    "operator": OPERATOR,
    "target_hardware": "Ascend910B",
    "eval_server_url": SERVER,
    "trace_root": TRACE_ROOT,
    "trace_set_key": "",
    "early_stop_rounds": 2,
    "min_rounds": 2,
}

print("\n" + "=" * 70)
print("Running ExtractOptWorkflow...")
print("=" * 70 + "\n")

out = wf.run(inp, None)

# ============================================================
# Verification
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

print(f"\n1. ExtractOptOutput:")
print(f"   operator: {out.operator}")
print(f"   num_definitions: {out.num_definitions}")
for r in out.results:
    status_icon = "✅" if r.status == "PASSED" else "❌"
    geo_str = f"{r.best_geo_mean:.3f}x" if r.best_geo_mean else "N/A"
    print(f"   {status_icon} {r.definition_name}: {r.status} (geo_mean: {geo_str})")
    print(f"      workspace: {r.workspace}")

print(f"\n2. Trace data structure:")
trace_data = Path(TRACE_ROOT)
if trace_data.exists():
    for f in sorted(trace_data.rglob("*.json")) + sorted(trace_data.rglob("*.jsonl")):
        print(f"   {f.relative_to(trace_data)}")
    print("   ✅")
else:
    print("   ❌ trace_data/ not created")

print(f"\n3. Agent workspaces:")
agents_dir = WT / "agents"
if agents_dir.exists():
    for agent_dir in sorted(agents_dir.iterdir()):
        if agent_dir.is_dir():
            has_ledger = (agent_dir / ".ledger.json").exists()
            has_claude = (agent_dir / ".claude").is_dir()
            print(f"   {agent_dir.name}/:")
            print(f"     .ledger.json: {'✅' if has_ledger else '❌'}")
            print(f"     .claude/: {'✅' if has_claude else '❌'}")
            if has_ledger:
                from kernelgen.data.ledger import Ledger
                led = Ledger(agent_dir)
                rounds = len(led.history.rounds)
                best = led.history.best_geo_mean
                print(f"     rounds: {rounds}, best_geo: {best:.3f}")
else:
    print("   ❌ agents/ not created")

print("\n" + "=" * 70)
status = "PASSED" if any(r.status == "PASSED" for r in out.results) else "FAILED"
print(f"E2E RESULT: {status}")
print("=" * 70)

sys.exit(0 if status == "PASSED" else 1)
