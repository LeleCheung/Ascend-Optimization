#!/usr/bin/env python3
"""根据已完成的分块诊断生成 v4；不修改已经测试的 v3 源码。"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = root / 'candidates/profiling-k128-pipeline-v3-20261010.py'
target = root / 'candidates/profiling-k256-dotacc-v4-20261010.py'
text = source.read_text(encoding='utf-8')
assert text.count('        acc += tl.dot(a, b, allow_tf32=False)') == 1
text = text.replace('        acc += tl.dot(a, b, allow_tf32=False)',
                    '        acc = tl.dot(a, b, acc, allow_tf32=False)')
old = '    bm, bn, bk = (32, 32, 32) if tiny else (64 if m <= 512 else 128, 128, 128)'
assert text.count(old) == 1
text = text.replace(old, old + '''
    # 真机分块诊断：低精度大矩阵增加 K 分块，减少循环和搬运同步。
    if input.dtype in (torch.float16, torch.bfloat16) and min(m, n, k) >= 1024:
        bk = 256''')
text = text.replace('    # ??????????????? Ascend ?????????',
                    '    # 保留原版的 ReLU 表达式，兼容当前 Ascend 编译链。')
text = text.replace('    # ??????????????????????? K128 ???',
                    '    # 非整齐形状使用原版，避免不规则边界的流水线问题。')
compile(text, str(target), 'exec')
assert not target.exists() or target.read_text(encoding='utf-8') == text
target.write_text(text, encoding='utf-8', newline='\n')
target.with_suffix('.provenance.json').write_text(json.dumps({
    'upstream_commit': 'd6a8eec473517a3d68157b208eb9c057eb1d4c50',
    'parent_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'candidate_sha256': hashlib.sha256(text.encode()).hexdigest(),
    'diagnostic': 'reports/diagnostics-20261010/matmul-tile-pipeline-20261010/artifacts/tile-results.json',
    'changes': ['使用 dot 的累加器参数', '连续整齐低精度大矩阵 K 分块 128→256'],
    'verification': '诊断阶段与 v3 最大绝对差为零；仍需完整 42 用例评测',
}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(target.name, hashlib.sha256(text.encode()).hexdigest())
