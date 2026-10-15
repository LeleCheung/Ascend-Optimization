"""补充原地/非连续/边界语义检查，不计入性能计时，不替代 KGS 合同。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('rad2deg_candidate', a.source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
rows = []
for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    for label, cpu in [
        ('empty', torch.empty(0)), ('one', torch.tensor([.1])),
        ('odd-1025', torch.linspace(-1, 1, 1025)),
        ('odd-2050', torch.linspace(-1, 1, 2050)),
        ('transpose', torch.linspace(-1, 1, 35).reshape(5, 7).t()),
        ('strided-slice', torch.linspace(-1, 1, 56).reshape(7, 8)[:, ::2]),
        ('special', torch.tensor([float('nan'), float('inf'), -float('inf'), 0., -0., 1., -1.])),
    ]:
        inp = cpu.to(device='npu:0', dtype=dtype)
        # 转移设备可能压紧步长，因此在设备上显式建立视图。
        if label == 'transpose':
            inp = cpu.t().to(device='npu:0', dtype=dtype).t()
        if label == 'strided-slice':
            base = torch.linspace(-1, 1, 56, device='npu:0', dtype=dtype).reshape(7, 8)
            inp = base[:, ::2]
        expected = torch.rad2deg(inp.detach().cpu().clone())
        pointer = inp.data_ptr()
        out = m.run(inp)
        torch.npu.synchronize()
        assert out is inp and out.data_ptr() == pointer, (label, '未返回原输入')
        tolerance = 2e-6 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(inp.cpu(), expected, rtol=tolerance, atol=tolerance, equal_nan=True)
        rows.append({'dtype': str(dtype), 'case': label, 'contiguous': inp.is_contiguous(), 'passed': True})
result = {'status': 'PASSED', 'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
          'cases': rows, 'scope': '补充语义检查；不替代KGS完整合同，不计性能'}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'PASSED', 'cases': len(rows)}))
