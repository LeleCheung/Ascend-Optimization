#!/usr/bin/env python3
"""保留 master 内核与 autotune，将超限的行网格拆成合法的多次启动。"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = root/'candidates/flaggems-master-dim-adapter-v2-20261010.py'
destination = root/'candidates/flaggems-master-grid-adapter-20261010.py'
old = '''        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            amin_kernel[grid](inp, out, M, N)
'''
new = '''        with torch_device_fn.device(inp.device):
            if M <= 65535:
                grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
                amin_kernel[grid](inp, out, M, N)
            else:
                # Ascend coreDim <= 65535；拆分行范围，原内核与调优配置不变。
                # 即使 autotune 选择 BLOCK_M=1，每次启动也不会超限。
                flat_input = inp.reshape(-1)
                flat_output = out.reshape(-1)
                for start in range(0, M, 65528):
                    rows = min(65528, M - start)
                    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]),)
                    amin_kernel[grid](flat_input[start * N:], flat_output[start:], rows, N)
'''
text = source.read_text(encoding='utf-8')
if text.count(old)!=1:raise SystemExit('源码启动位置不唯一，停止生成')
text = text.replace(old,new)
compile(text,str(destination),'exec')
if destination.exists() and destination.read_text(encoding='utf-8')!=text:
    raise SystemExit('候选冲突，拒绝覆盖')
destination.write_text(text,encoding='utf-8',newline='\n')
provenance = {
    'upstream_commit':'d6a8eec473517a3d68157b208eb9c057eb1d4c50',
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'candidate_sha256':hashlib.sha256(text.encode()).hexdigest(),
    'changes':['整数 dim 转列表、导出 run（继承入口适配）',
               '输出行数超过 65535 时，保留原内核并按最多 65528 行分次启动'],
    'unchanged':['master 归约内核','autotune 配置','dim_compress 转置','完整正确性与性能用例','计时方法'],
    'baseline_label':'FlagGems master 加启动兼容修复；不是未修改的上游性能',
}
destination.with_suffix('.provenance.json').write_text(
    json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(destination.name,provenance['candidate_sha256'])
