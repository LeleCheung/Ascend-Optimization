#!/usr/bin/env python3
"""将双 N tile 点积直接累计，并启用已验证的 Ascend 多缓冲流水。"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "candidates/profiling-dual-dispatch-v8-20261010.py"
target = root / "candidates/profiling-dual-acc-v9-20261012.py"
text = source.read_text(encoding="utf-8")
old = "            out.stride(0), out.stride(1), 128, 128, 256)"
new = "            out.stride(0), out.stride(1), 128, 128, 256,\n            multibuffer=True, unit_flag=True, sync_solver=False)"
if text.count(old) != 1:
    raise SystemExit("未找到唯一的 v8 双 tile 启动点")
result = text.replace(old, new)
result = result.replace("acc0 += tl.dot(a, b0, allow_tf32=False)", "acc0 = tl.dot(a, b0, acc0, allow_tf32=False)")
result = result.replace("acc1 += tl.dot(a, b1, allow_tf32=False)", "acc1 = tl.dot(a, b1, acc1, allow_tf32=False)")
target.write_bytes(result.encode('utf-8'))
(target.with_suffix(".provenance.json")).write_text(json.dumps({
    "parent": str(source.name),
    "parent_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "candidate_sha256": hashlib.sha256(result.encode()).hexdigest(),
    "change": "双 N tile 路径使用 dot 累加器，启用 multibuffer=True、unit_flag=True、sync_solver=False；探针中 fp16/bf16 1024/2048/4096 与 v8 数值一致，2048/4096 明显提速",
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(target, hashlib.sha256(result.encode()).hexdigest())
