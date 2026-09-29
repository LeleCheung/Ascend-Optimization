#!/usr/bin/env python
"""Run optimize_loop example — iterative kernel optimization using callable + run().

Each iteration:
1. run(optimize_agent) — CC reads workspace history, picks one optimization direction, outputs result
2. callable saves kernel version + updates PERFORMANCE.md in workspace
3. Next iteration's CC sees updated history

Usage:
    python kgrunner/examples/optimize_loop/run_example.py --input ops.jsonl
    python kgrunner/examples/optimize_loop/run_example.py --input ops.jsonl --gpu-ids 4 5
    python kgrunner/examples/optimize_loop/run_example.py --input ops.jsonl --max-iters 10 --target-speedup 1.5

Input JSONL format:
    {"OPERATOR": "softmax"}
    {"OPERATOR": "layernorm"}
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from kgrunner import run, run_parallel, load_agent, GPUPool

AGENT_DIR = Path(__file__).resolve().parent
optimize_agent = load_agent(AGENT_DIR)


def save_version(workspace: Path, iteration: int, result: dict):
    """Save kernel version and metadata to workspace/versions/vN/."""
    version_dir = workspace / "versions" / f"v{iteration + 1}"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta = {
        "iteration": iteration + 1,
        "speedup": result.get("speedup"),
        "test_passed": result.get("test_passed"),
        "optimization_direction": result.get("optimization_direction", ""),
        "status": result.get("status"),
    }
    (version_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Save kernel code if agent reported a kernel path
    kernel_path = result.get("kernel_path")
    if kernel_path:
        src = workspace / kernel_path
        if src.exists():
            shutil.copy2(src, version_dir / "kernel.py")


def update_performance_md(workspace: Path):
    """Regenerate PERFORMANCE.md from all saved versions."""
    versions_dir = workspace / "versions"
    if not versions_dir.exists():
        return

    rows = []
    best_version = None
    best_speedup = 0.0

    for vdir in sorted(versions_dir.iterdir(), key=lambda p: p.name):
        meta_path = vdir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        v = vdir.name
        status = "PASS" if meta.get("test_passed") else "FAIL"
        speedup = meta.get("speedup")
        speedup_str = f"{speedup:.4f}" if speedup is not None else "-"
        direction = meta.get("optimization_direction", "")[:50]
        rows.append(f"| {v} | {status} | {speedup_str} | {direction} |")

        if meta.get("test_passed") and speedup is not None and speedup > best_speedup:
            best_speedup = speedup
            best_version = v

    md = f"""# Performance History

Best: {best_version or 'none'} (speedup: {best_speedup:.4f})

| Version | Status | Speedup | Direction |
|---------|--------|---------|-----------|
"""
    md += "\n".join(rows) + "\n"
    (workspace / "PERFORMANCE.md").write_text(md)


def make_optimize_loop(max_iters: int, target_speedup: float):
    """Create an optimize_loop callable with the given parameters."""

    def optimize_loop(input_data: dict, gpu_id=None, workspace=None):
        result = {}
        for i in range(max_iters):
            result = run(
                optimize_agent,
                input_data,
                gpu=gpu_id,
                workspace=workspace,
            )
            speedup = result.get("speedup", 0)
            print(f"  [{input_data['OPERATOR']}] iter {i+1}/{max_iters}: speedup={speedup:.4f}")

            # Save version and update history for next iteration
            if workspace:
                save_version(Path(workspace), i, result)
                update_performance_md(Path(workspace))

            if speedup >= target_speedup:
                print(f"  [{input_data['OPERATOR']}] target reached!")
                break

        return result

    # Carry agent metadata so run_parallel() can auto-resolve workspace/resources
    optimize_loop.workspace_config = optimize_agent.workspace_config
    optimize_loop.resources = optimize_agent.resources
    optimize_loop.name = optimize_agent.name

    return optimize_loop


def main():
    parser = argparse.ArgumentParser(description="optimize_loop: iterative kernel optimization via kgrunner")
    parser.add_argument("--input", type=Path, required=True, help="JSONL file, one task per line")
    parser.add_argument("--gpu-ids", type=int, nargs="*", help="GPU device IDs, e.g. --gpu-ids 0 1 2 3 (default: auto-detect all)")
    parser.add_argument("--max-iters", type=int, default=20, help="Max optimization iterations per operator (default: 20)")
    parser.add_argument("--target-speedup", type=float, default=0.8, help="Stop when speedup >= target (default: 0.8)")
    args = parser.parse_args()

    pool = GPUPool(device_ids=args.gpu_ids)
    optimize_loop = make_optimize_loop(args.max_iters, args.target_speedup)

    results = run_parallel(
        optimize_loop, args.input,
        gpu=pool,
        workspace_name_key="OPERATOR",
    )

    for result in results:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
