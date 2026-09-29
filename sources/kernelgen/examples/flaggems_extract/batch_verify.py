#!/usr/bin/env python
"""Batch-extract + self-verify FlagGems operators on the eval server.

Runs the FlagGemsExtractWorkflow for each operator. The extractor agent
self-verifies each definition via the eval_only MCP tool (reference-as-solution)
before emitting; this driver records pass/fail and only the verified definitions
land in the output trace root.

Must run INSIDE the 910B-6 docker (needs the eval server + FlagGems + dpsk):

    cd /data/xuyao/kernelgen && source env.sh
    export PYTHONPATH=/data/xuyao:/data/xuyao/flashinfer-bench
    export FIB_EVAL_SERVER=http://127.0.0.1:18081 FIB_TARGET_HW=Ascend910B
    python3 examples/flaggems_extract/batch_verify.py \
        --kernel-list /data/xuyao/kernel_list \
        --flaggems-repo /data/xuyao/FlagGems-master \
        --trace-root /data/xuyao/flaggems-v2-out \
        --report /data/xuyao/flaggems-v2-out/_report.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# repo parent (/data/xuyao) so `import kernelgen` works
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.workflows.flaggems_extract import FlagGemsExtractWorkflow
from kernelgen.agents.extractor.flaggems import (
    implemented_operators,
    OperatorNotImplementedError,
)

KERNELGEN_ROOT = Path(__file__).resolve().parents[2]


def _read_ops(args) -> list[str]:
    if args.ops:
        return [o.strip() for o in args.ops.split(",") if o.strip()]
    ops = []
    for line in Path(args.kernel_list).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and s != "算子名称":
            ops.append(s)
    if args.limit:
        ops = ops[: args.limit]
    return ops


def _make_runtime_factory(model, timeout):
    def make_rt(path):
        # allowed_tools MUST include Write (scratch files) + the eval_only MCP tool.
        return ClaudeRuntime(
            workspace=path,
            model=model,
            allowed_tools="Bash,Read,Write,Glob,Grep,mcp__kernelgen__eval_only",
            permission_mode="acceptEdits",
            timeout=timeout,
            idle_timeout=max(120, timeout // 2),
        )
    return make_rt


def _run_one(op, args, make_rt) -> dict:
    """Extract+verify one operator. Returns a report row."""
    verify_root = f"/tmp/flaggems_verify_{op}"
    wf = FlagGemsExtractWorkflow(cwd=str(KERNELGEN_ROOT), runtime_factory=make_rt)
    t0 = time.time()
    try:
        result = wf.run({
            "operator": op,
            "flaggems_repo": args.flaggems_repo,
            "trace_root": args.trace_root,
            "verify_trace_root": verify_root,
            "verify_target_hw": os.environ.get("FIB_TARGET_HW", "Ascend910B"),
        })
        rd = result.model_dump()
        defs = rd.get("definitions", [])
        if not defs:
            return {"operator": op, "status": "NO_DEFINITION",
                    "seconds": round(time.time() - t0, 1)}
        # AUTHORITATIVE gate: the driver itself re-evaluates each emitted definition
        # (reference-as-solution) on the server — never trust the agent's self-report.
        verified = {name: _driver_verify(args.trace_root, name) for name in defs}
        all_ok = all(v["status"] == "PASSED" for v in verified.values())
        return {
            "operator": op, "status": "PASSED" if all_ok else "VERIFY_FAILED",
            "definitions": defs, "verified": verified,
            "seconds": round(time.time() - t0, 1),
        }
    except OperatorNotImplementedError as e:
        return {"operator": op, "status": "SKIPPED_NOT_IMPLEMENTED",
                "error": str(e)[:200], "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"operator": op, "status": "ERROR",
                "error": f"{type(e).__name__}: {e}"[:300],
                "seconds": round(time.time() - t0, 1)}


def _driver_verify(trace_root, name) -> dict:
    """Authoritatively re-evaluate one emitted definition on the server.

    Copies ONLY this op's definition + phased_workloads into an ISOLATED temp
    trace dir (so one malformed op can't poison another op's TraceSet load), writes
    the definition's own reference (contains run(); gen_inputs/valid ignored on the
    solution side) as main.py, and calls evaluate_only. Returns {status, geo_mean?, log?}.
    """
    import shutil
    from kernelgen.tools.eval_only import evaluate_only

    hits = list(Path(trace_root).glob(f"definitions/*/{name}.json"))
    if not hits:
        return {"status": "ERROR", "log": "definition file not found"}
    # Prefer the NEWEST match: an op re-run under a different op_type dir (e.g. a
    # rename conv -> convolution) leaves a stale older duplicate that must not win.
    def_json = max(hits, key=lambda p: p.stat().st_mtime)
    op_type = def_json.parent.name
    try:
        defn = json.loads(def_json.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"status": "ERROR", "log": f"bad definition json: {e}"}

    # Isolated single-op trace dir.
    iso = Path(f"/tmp/flaggems_drvverify_{name}")
    if iso.exists():
        shutil.rmtree(iso)
    (iso / "definitions" / op_type).mkdir(parents=True, exist_ok=True)
    (iso / "phased_workloads" / op_type).mkdir(parents=True, exist_ok=True)
    shutil.copy2(def_json, iso / "definitions" / op_type / def_json.name)
    for phase in ("correctness", "timing"):
        src = Path(trace_root) / "phased_workloads" / op_type / f"{name}.{phase}.jsonl"
        if src.is_file():
            shutil.copy2(src, iso / "phased_workloads" / op_type / src.name)

    main_py = iso / "main.py"
    main_py.write_text(defn.get("reference", ""), encoding="utf-8")

    res = evaluate_only(
        kernel_path=str(main_py),
        definition=name,
        trace_root=str(iso),
        target_hardware=os.environ.get("FIB_TARGET_HW", "Ascend910B"),
    )
    out = {"status": res.get("status", "ERROR")}
    if res.get("geo_mean") is not None:
        out["geo_mean"] = round(res["geo_mean"], 4)
    if out["status"] != "PASSED":
        out["log"] = str(res.get("log", res.get("error", "")))[:300]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel-list", default="/data/xuyao/kernel_list")
    p.add_argument("--ops", default="", help="comma-separated ops (overrides --kernel-list)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--flaggems-repo", default="/data/xuyao/FlagGems-master")
    p.add_argument("--trace-root", default="/data/xuyao/flaggems-v2-out")
    p.add_argument("--report", default="")
    p.add_argument("--model", default=os.environ.get("MODEL", "inherit"))
    p.add_argument("--timeout", type=int, default=1200)
    args = p.parse_args()

    ops = _read_ops(args)
    impl = implemented_operators(args.flaggems_repo)
    make_rt = _make_runtime_factory(args.model, args.timeout)

    print(f"batch: {len(ops)} ops -> {args.trace_root}  (model={args.model})", flush=True)
    rows = []
    for i, op in enumerate(ops, 1):
        gate = "impl" if op in impl else "NOT_IN_ALL"
        print(f"\n[{i}/{len(ops)}] {op}  ({gate})", flush=True)
        row = _run_one(op, args, make_rt)
        rows.append(row)
        print(f"    -> {row['status']}  {row.get('definitions', row.get('error',''))}"
              f"  ({row['seconds']}s)", flush=True)
        # incremental report so a crash mid-batch keeps progress
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                         encoding="utf-8")

    passed = [r for r in rows if r["status"] == "PASSED"]
    print(f"\n{'='*60}\nDONE: {len(passed)}/{len(rows)} PASSED")
    for r in rows:
        if r["status"] != "PASSED":
            print(f"  {r['status']:24s} {r['operator']}")
    if args.report:
        print(f"report: {args.report}")
    return 0 if len(passed) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
