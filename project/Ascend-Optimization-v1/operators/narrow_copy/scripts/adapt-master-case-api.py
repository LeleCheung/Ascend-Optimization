#!/usr/bin/env python3
"""为固定 master 的 narrow_copy benchmark 增加懒加载 case API，不改变算子或用例。"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

MASTER = "d6a8eec473517a3d68157b208eb9c057eb1d4c50"
BENCHMARK = "benchmark/test_narrow_copy.py"
SOURCE = "src/flag_gems/ops/narrow_copy.py"

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("checkout", type=Path)
p.add_argument("--provenance", type=Path, required=True)
a = p.parse_args()

def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(a.checkout), *args])

if git("rev-parse", "HEAD").decode().strip() != MASTER:
    raise SystemExit("仅允许从固定 master 创建新的独立适配副本")
if git("status", "--porcelain=v1", "--", "*.py", "pyproject.toml", "pytest.ini", "setup.cfg").strip():
    raise SystemExit("副本源码不干净，拒绝覆盖")

original = git("show", f"{MASTER}:{BENCHMARK}")
text = original.decode("utf-8")
anchor = "\ndef narrow_copy_gbps(bench_fn_args, latency):"
functions = '''
def narrow_copy_case_fn(shape, cur_dtype):
    dim = 0
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"dim": dim, "start": shape[dim] // 4, "length": shape[dim] // 2},
        builder_args=(tuple(shape),),
    )


def narrow_copy_build_inputs(plan, cur_dtype, device):
    return next(narrow_copy_input_fn(plan.builder_args[0], cur_dtype, device))

'''
if text.count(anchor) != 1 or text.count("input_fn=narrow_copy_input_fn,") != 1:
    raise SystemExit("上游结构不符合固定快照，拒绝自动适配")
text = text.replace(anchor, "\n" + functions + anchor)
text = text.replace("input_fn=narrow_copy_input_fn,",
                    "case_fn=narrow_copy_case_fn,\n        build_inputs_fn=narrow_copy_build_inputs,")
compile(text, BENCHMARK, "exec")
adapted = text.encode("utf-8")
(a.checkout / BENCHMARK).write_bytes(adapted)
source = git("show", f"{MASTER}:{SOURCE}")
if (a.checkout / SOURCE).read_bytes() != source:
    raise SystemExit("算子源码与 master 不一致")
a.provenance.parent.mkdir(parents=True, exist_ok=True)
a.provenance.write_text(json.dumps({
    "upstream_commit": MASTER, "benchmark_path": BENCHMARK,
    "original_benchmark_sha256": hashlib.sha256(original).hexdigest(),
    "adapted_benchmark_sha256": hashlib.sha256(adapted).hexdigest(),
    "operator_path": SOURCE, "operator_sha256": hashlib.sha256(source).hexdigest(),
    "change": "只增加 case 枚举；shape 筛选、精度、dim/start/length、原输入生成器与计时保持不变",
    "next_step": "在该独立副本提交 benchmark 适配，固定 commit 后启动专用 KGS",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("已生成 benchmark case API；算子源码保持固定 master 内容")
