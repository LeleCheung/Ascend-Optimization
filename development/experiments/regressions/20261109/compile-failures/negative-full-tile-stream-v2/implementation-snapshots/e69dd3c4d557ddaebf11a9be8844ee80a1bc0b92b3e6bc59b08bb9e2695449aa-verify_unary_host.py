"""核验宿主缓存的实际 Triton 编译产物、动态输入地址及冷启动和热启动语义。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-stream-tail', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu
    from triton.runtime.jit import JITFunction
    from torch.utils._python_dispatch import TorchDispatchMode

    class CandidateOnly(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = []

        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            name = str(function)
            self.operations.append(name)
            allowed = {'aten.empty', 'aten.empty_like', 'aten.empty_strided', 'aten.alias',
                       'aten.detach', 'aten.view', 'aten.as_strided', 'aten.slice', 'aten.select'}
            if '.'.join(name.split('.')[:2]) not in allowed:
                raise RuntimeError('候选调用了非元数据ATen入口：'+name)
            return function(*args, **(kwargs or {}))
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('_unary_host_candidate', args.source)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    original = JITFunction.run
    entries = {}

    def capture(kernel, *values, **options):
        compiled = original(kernel, *values, **options)
        if compiled is not None and hasattr(compiled, 'asm'):
            symbol = kernel.__name__
            identity = (symbol, str(compiled.metadata))
            if identity not in entries:
                folder = args.output / ('kernel-' + str(len(entries)))
                folder.mkdir()
                artifacts = []
                for kind, value in compiled.asm.items():
                    if not isinstance(value, (str, bytes)):
                        continue
                    target = folder / ('kernel.' + kind)
                    target.write_bytes(value.encode() if isinstance(value, str) else value)
                    artifacts.append(dict(kind=kind, path=str(target), sha256=sha(target)))
                assert {'ttir', 'npubin'}.issubset({a['kind'] for a in artifacts})
                entries[identity] = dict(symbol=symbol, metadata=str(compiled.metadata), artifacts=artifacts)
        return compiled

    rows = []
    JITFunction.run = capture
    try:
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            workloads = [(1024, 'aligned'), (1031, 'masked'), (1048576, 'streaming'),
                         (1024, 'strided'), (1024, 'unaligned')]
            if args.include_stream_tail:
                workloads.append((1048579, 'streaming-tail'))
            for count, layout in workloads:
                # 每次重新分配输入；第二次调用使用新stream，避免缓存旧地址或旧stream。
                for attempt in range(2):
                    stream = torch.npu.Stream()
                    with torch.npu.stream(stream):
                        if layout == 'strided':
                            base = torch.randn(count * 2, dtype=dtype, device='npu:0')
                            tensor = base[::2]
                        elif layout == 'unaligned':
                            base = torch.randn(count + 1, dtype=dtype, device='npu:0')
                            tensor = base[1:]
                        else:
                            tensor = torch.randn(count, dtype=dtype, device='npu:0')
                        expected = -tensor.cpu()
                        mode = CandidateOnly()
                        with mode:
                            result = candidate.run(tensor)
                    stream.synchronize()
                    assert result is tensor
                    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
                    rows.append(dict(dtype=str(dtype), count=count, layout=layout,
                                     attempt=attempt, passed=True, aten_operations=mode.operations))
    finally:
        JITFunction.run = original
    symbols = {e['symbol'] for e in entries.values()}
    assert symbols == {'_negative_nomask', '_negative_masked', '_negative_streaming', '_negative_strided'}, symbols
    stats = dict(candidate._BRIDGE.stats())
    assert stats['fast_calls'] > 0 and stats['fallback_calls'] > 0
    result = dict(status='verified', source_sha256=sha(args.source),
                  checker_sha256=sha(__file__), cases=rows, entries=list(entries.values()),
                  host_cache=stats,
                  scope='实际运行捕获的Triton编译对象；与逐路径源码审查及冻结语义检查共同使用，不替代硬件任务trace')
    (args.output/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(status=result['status'], cases=len(rows), kernels=sorted(symbols), host_cache=stats)))


if __name__ == '__main__':
    main()
