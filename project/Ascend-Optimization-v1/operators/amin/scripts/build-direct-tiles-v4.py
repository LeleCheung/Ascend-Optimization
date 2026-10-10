#!/usr/bin/env python3
"""使用已完成的真机分块筛选生成 v4，保留已测 v3。"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = root / 'candidates/direct-axis-global-v3-20261010.py'
target = root / 'candidates/direct-axis-tiles-v4-20261010.py'
text = source.read_text(encoding='utf-8')
old = '''            bo = min(4, max(1, 2048 // br))
            amin_direct_last_kernel'''
new = '''            bo = min(4, max(1, 2048 // br))
            # 大尺寸末轴采用已在真机逐值核对的多行分块。
            if outer >= 1024 and reduction >= 1024:
                bo, br = (16, 512) if inp.dtype == torch.float16 else (16, 256)
                if inp.dtype == torch.bfloat16:
                    bo, br = 32, 256
            amin_direct_last_kernel'''
assert text.count(old) == 1
text = text.replace(old, new)
old = '''            br = min(128, triton.next_power_of_2(reduction))
            amin_direct_middle_kernel'''
new = '''            br = min(128, triton.next_power_of_2(reduction))
            # 32 行归约片减少 UB 压力，扩大连续列提高搬运粒度。
            if reduction >= 512 and inner >= 128:
                bc, br = (128, 32) if inp.dtype == torch.float32 else (256, 32)
            amin_direct_middle_kernel'''
assert text.count(old) == 1
text = text.replace(old, new)
old = '''    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, float('inf'))
    result = tl.min(values.to(tl.float32), axis=0)
    tl.store(Y + tl.program_id(0), result)'''
new = '''    # 超大全归约也限制 coreDim；一个 program 可处理多个独立分块。
    for chunk in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = chunk * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(X + offsets, offsets < N, float('inf'))
        result = tl.min(values.to(tl.float32), axis=0)
        tl.store(Y + chunk, result)'''
assert text.count(old) == 1
text = text.replace(old, new).replace(
    'amin_global_chunk_kernel[(chunks,)]', 'amin_global_chunk_kernel[(min(chunks, 65535),)]')
compile(text, str(target), 'exec')
assert not target.exists() or target.read_text(encoding='utf-8') == text
target.write_text(text, encoding='utf-8', newline='\n')
target.with_suffix('.provenance.json').write_text(json.dumps({
    'parent_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'candidate_sha256': hashlib.sha256(text.encode()).hexdigest(),
    'diagnostic': 'reports/diagnostics-20261010/amin-direct-tiles-20261010/artifacts/direct-tiles.json',
    'changes': ['大末轴按 dtype 选择多行分块', '中间轴扩大连续列并减小归约片',
                '全归约启动数限制为 65535，超限分块由 program 内循环处理'],
    'verification': '分块筛选逐值一致；最终验收仍需完整 27 项评测',
}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(target.name, hashlib.sha256(text.encode()).hexdigest())
