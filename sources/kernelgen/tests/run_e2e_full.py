"""E2E test for the new kernelgen package: single-epoch, 2 agents, verifies all modules.

Validates:
1. AnalyzerAgent (cold-start analysis)
2. CoderAgent × 2 (parallel optimization, uses eval_round + finalize_round finalization)
3. DistillerAgent (trajectory → candidate experience)
4. Epoch knowledge reducer (score-aware N-way MergeAgent)
5. EpochSummaryAgent (cross-agent comparison)
6. KB writeback (candidate reduction → git commit)
7. IsolatedDirectory (workspace isolation + .claude + history-free KB snapshot)

Uses deepseek-v4-pro via zyapi + 910B eval server.

    cd /data/akg_kernel_bench_lite
    PYTHONPATH=. python -u kernelgen/tests/run_e2e_full.py
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Ensure we can import kernelgen
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernelgen.framework import (
    ClaudeRuntime,
    IsolatedDirectory,
    copy_claude_directory,
    materialize_mcp_configuration,
)
from kernelgen.workflows.optimization.kernelgen import KernelGenWorkflow, KernelGenInput
from kernelgen.data.ledger import Ledger
from kernelgen.data.stop_policy import StopConfig
from kernelgen.data.catalog import (
    DEFAULT_CATALOG_NAME,
    resolve_builtin_catalog_path,
)
from kernelgen.data.trace import load_catalog_optimization_context

# Config
REPO = Path(os.environ.get("FIB_REPO", "/data/akg_kernel_bench_lite/flashinfer-bench"))
KERNELGEN_DIR = Path(os.environ.get("KERNELGEN_DIR", "/data/akg_kernel_bench_lite/kernelgen"))
WT = Path(os.environ.get("WT", "/tmp/kernelgen_e2e"))
DEFN_NAME = os.environ.get("DEFN", "flaggems_rsqrt")
SERVER = os.environ.get("FIB_EVAL_SERVER", "http://localhost:8000")
CATALOG_NAME = os.environ.get(
    "KERNELGEN_CATALOG_NAME",
    DEFAULT_CATALOG_NAME,
)
MODEL = os.environ.get("MODEL", "deepseek-v4-pro[1m]")
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://zyapi.xmsxb.com")
AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")

# Clean
if WT.exists():
    shutil.rmtree(WT)
WT.mkdir(parents=True)

print("=" * 70)
print(f"KernelGen E2E (new package): {DEFN_NAME}, model={MODEL}")
print(f"Workspace: {WT}")
print(f"Eval server: {SERVER}")
print("=" * 70)

# ============================================================
# Setup workflow workspace
# ============================================================

# Materialize provider agents/skills and shared MCP registration from repo.
copy_claude_directory(KERNELGEN_DIR / ".claude", WT / ".claude")
print("✅ Materialized neutral roles, skills, and provider settings")
mcp_source = KERNELGEN_DIR / ".kernelgen" / "mcp.json"
if mcp_source.is_file():
    materialize_mcp_configuration(mcp_source, WT)
    print("✅ Materialized neutral MCP configuration")

# Create base KB (empty for this test); the reducer creates the typed entry.
(WT / "kb").mkdir(parents=True)

# Git init base KB
env_git = {**os.environ, "GIT_AUTHOR_NAME": "init", "GIT_AUTHOR_EMAIL": "i@i",
            "GIT_COMMITTER_NAME": "init", "GIT_COMMITTER_EMAIL": "i@i"}
subprocess.run(["git", "init"], cwd=str(WT / "kb"), capture_output=True, check=True)
subprocess.run(["git", "add", "."], cwd=str(WT / "kb"), capture_output=True, check=True)
subprocess.run(["git", "commit", "-m", "base", "--allow-empty"],
               cwd=str(WT / "kb"), capture_output=True, check=True, env=env_git)
print("✅ Base KB created + git initialized")

# ============================================================
# Load definition from the Server-owned Catalog
# ============================================================
definition, _ = load_catalog_optimization_context(
    resolve_builtin_catalog_path(CATALOG_NAME),
    DEFN_NAME,
)

print(f"✅ Definition loaded: {definition.name} ({definition.op_type})")

# ============================================================
# Set env for tools (PYTHONPATH so eval/finalize tools can import)
# ============================================================
os.environ["PYTHONPATH"] = f"{REPO}:{REPO / 'examples'}:{KERNELGEN_DIR.parent}"
os.environ["FIB_EVAL_SERVER"] = SERVER

# ============================================================
# Build input
# ============================================================
inp = {
    "definition": definition.model_dump(),
    "target_hardware": "Ascend910B",
    "n_parallel": int(os.environ.get("N_PARALLEL", "2")),
    "n_epoch": int(os.environ.get("N_EPOCH", "1")),   # default single epoch; override via N_EPOCH
    "eval_server_url": SERVER,
    "catalog_name": CATALOG_NAME,
    "profile_enabled": os.environ.get("PROFILE_ENABLED", "1") not in ("0", "false", "False", ""),
    "timeout": int(os.environ.get("E2E_TIMEOUT", "3600")),
    "early_stop_rounds": 2,
    "min_rounds": 2,
}

# ============================================================
# Build workflow
# ============================================================
def make_rt(path):
    return ClaudeRuntime(
        workspace=path,
        model=MODEL,
        base_url=BASE_URL,
        auth_token=AUTH_TOKEN,
        timeout=3600,
        idle_timeout=1800,
    )

wf = KernelGenWorkflow(cwd=str(WT), runtime_factory=make_rt)

print("\n" + "=" * 70)
print("Running KernelGenWorkflow...")
print("=" * 70 + "\n")

out = wf.run(inp, None)

# ============================================================
# Verification
# ============================================================
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

print(f"\n1. KernelGenOutput:")
print(f"   status: {out.status}")
print(f"   best_geo_mean: {out.best_geo_mean}")
print(f"   num_agents: {out.num_agents}")
print(f"   per_agent: {out.per_agent}")
print(f"   best_code: {len(out.best_code)} chars")

print(f"\n2. Epoch directories:")
epoch_dir = WT / "1R"
if epoch_dir.exists():
    agents = [d.name for d in epoch_dir.iterdir() if d.is_dir()]
    print(f"   1R/: {agents} ✅")
else:
    print(f"   1R/: NOT FOUND ❌")

print(f"\n3. Agent workspaces (checking first agent):")
agent0_dir = epoch_dir / "agent0" if epoch_dir.exists() else None
if agent0_dir and agent0_dir.exists():
    has_ledger = (agent0_dir / ".ledger.json").exists()
    has_stop_config = (agent0_dir / ".stop_config.json").exists()
    has_new_exp = (agent0_dir / ".new_experience.md").exists()
    has_claude = (agent0_dir / ".claude").is_dir()
    has_mcp = (
        (agent0_dir / ".kernelgen" / "mcp.json").is_file()
        and (agent0_dir / ".mcp.json").is_file()
    )
    has_kb = (agent0_dir / "kb").is_dir()
    print(f"   .ledger.json: {'✅' if has_ledger else '❌'}")
    print(f"   .stop_config.json: {'✅' if has_stop_config else '❌'}")
    print(f"   .new_experience.md: {'✅' if has_new_exp else '❌'}")
    print(f"   .claude/: {'✅' if has_claude else '❌'}")
    print(f"   MCP config: {'✅' if has_mcp else '❌'}")
    print(f"   kb/: {'✅' if has_kb else '❌'}")
    print(f"   kb snapshot has no .git: {'✅' if not (agent0_dir / 'kb' / '.git').exists() else '❌'}")
    if has_ledger:
        led = Ledger(agent0_dir)
        print(f"   ledger: {len(led.history.rounds)} rounds, best={led.history.best_geo_mean:.3f}")
    if has_new_exp:
        exp = (agent0_dir / ".new_experience.md").read_text()
        print(f"   .new_experience.md: {len(exp)} chars")
        print(f"   preview: {exp[:150]}...")
else:
    print(f"   agent0 dir NOT FOUND ❌")

print(f"\n4. KB git log (base_kb after merge):")
r = subprocess.run(["git", "log", "--oneline"], cwd=str(WT / "kb"), capture_output=True, text=True)
for line in r.stdout.strip().splitlines():
    print(f"   {line}")
has_merge_commit = "epoch 1" in r.stdout
print(f"   KB merge commit: {'✅' if has_merge_commit else '❌ (no agent KB changes detected)'}")

print(f"\n5. KB content after merge:")
exp_file = WT / "kb" / "experience" / "by_definition" / d.op_type / DEFN_NAME / "Ascend910B" / "experience.md"
if exp_file.exists():
    content = exp_file.read_text()
    if "Prior: No experience" in content and len(content) < 100:
        print(f"   ❌ KB unchanged (no merge happened)")
    else:
        print(f"   ✅ KB updated ({len(content)} chars)")
        print(f"   preview: {content[:200]}...")

print("\n" + "=" * 70)
print("E2E DONE")
print("=" * 70)
