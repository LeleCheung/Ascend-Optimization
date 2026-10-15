#!/usr/bin/env python3
"""在独立进程中阻止 PyTorch 复制 API，验证指定候选的小型连续 CANN DMA 路径。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('输出已存在，请保留原始核验记录')
    import torch
    import torch_npu
    spec = importlib.util.spec_from_file_location('dma_candidate', a.source)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    cpu = torch.arange(128, dtype=torch.float32).reshape(16, 8)
    expected = cpu[3:8].clone()
    inp = cpu.to(a.device)
    runtime = candidate._AclRuntime.instance()
    original = runtime.memcpy_async
    if original is None:
        raise RuntimeError('未加载 CANN memcpy，不能核验直接调用路径')
    calls, saved = [], []

    def observed(*args):
        ret = original(*args)
        calls.append({'destMax': args[1].value, 'count': args[3].value,
                      'kind': args[4].value, 'return_code': ret})
        return ret

    def blocked(*args, **kwargs):
        raise AssertionError('候选命中了 PyTorch 复制 API')

    runtime.memcpy_async = observed
    for owner, name, label in [(torch.Tensor, 'copy_', 'Tensor.copy_'),
                               (torch, 'narrow_copy', 'torch.narrow_copy'),
                               (torch.Tensor, 'narrow_copy', 'Tensor.narrow_copy')]:
        if hasattr(owner, name):
            saved.append((owner, name, getattr(owner, name), label))
            setattr(owner, name, blocked)
    try:
        out = candidate.run(inp, 0, 3, 5)
        torch.npu.synchronize(inp.device)
    finally:
        for owner, name, value, _ in saved:
            setattr(owner, name, value)
        runtime.memcpy_async = original
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert out.data_ptr() != inp.data_ptr() and len(calls) == 1
    assert calls[0] == {'destMax': 160, 'count': 160, 'kind': 3, 'return_code': 0}
    result = {'status': 'PASSED', 'scope': '小型连续单段快路径',
              'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
              'runtime_library': runtime.lib._name, 'symbol': 'aclrtMemcpyAsync',
              'blocked_pytorch_copy_apis': [label for _, _, _, label in saved],
              'calls': calls, 'independent_output': True, 'shape': list(out.shape),
              'correctness': 'exact_cpu_reference'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
