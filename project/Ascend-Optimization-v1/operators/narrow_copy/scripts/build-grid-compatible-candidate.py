#!/usr/bin/env python3
"""保留 master 的索引与分块，拆分超过 8192 program 的独立启动。"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
parent = root / 'candidates/flaggems-master-20261010.py'
text = parent.read_text(encoding='utf-8')
text = text.replace('    BLOCK_SIZE: tl.constexpr,\n', '    BLOCK_SIZE: tl.constexpr,\n    BASE: tl.constexpr = 0,\n', 1)
text = text.replace('    block_start = pid * BLOCK_SIZE\n', '    block_start = BASE + pid * BLOCK_SIZE\n', 1)
start = text.index('    narrow_copy_kernel[grid](\n')
end = text.index('\n    return out', start)
text = text[:start] + '''    # 只拆启动范围，沿用原内核、动态除数、1024 分块和输入/输出地址。
    with torch.npu.device(inp.device):
        for base in range(0, total_elements, 8192 * BLOCK_SIZE):
            programs = min(8192, triton.cdiv(total_elements - base, BLOCK_SIZE))
            narrow_copy_kernel[(programs,)](
                out, inp, total_elements, inp_total_elements, dim_size,
                dim_prod_post, start, length, BLOCK_SIZE=BLOCK_SIZE, BASE=base,
            )
''' + text[end:]
target = root / 'candidates/flaggems-master-grid-compatible-20261010.py'
compile(text, str(target), 'exec')
assert not target.exists() or target.read_bytes() == text.encode()
target.write_bytes(text.encode())
target.with_suffix('.provenance.json').write_text(json.dumps({
    'upstream_commit': 'd6a8eec473517a3d68157b208eb9c057eb1d4c50',
    'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
    'candidate_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    'change': '只拆分大输入 program 启动；保留 master 索引计算、1024 分块和完整测试；每次最多 8192 program',
    'verification': '待完整复验；必须标为 master 加启动兼容修复，不可称未修改 master',
}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(target.name)
