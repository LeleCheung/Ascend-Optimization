"""单CPU交叉编译expm1完整块提案，阻止设备初始化、加载和提交。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    raise RuntimeError('离线编译禁止设备初始化、加载或提交')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), '编译原件不覆盖'
    args.output.mkdir(parents=True)
    (args.output / ('checker-' + sha(__file__) + '.py')).write_bytes(Path(__file__).read_bytes())
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TRITON_ASCEND_ARCH'] = 'Ascend910B4'
    os.environ['TRITON_CACHE_DIR'] = str(args.output / 'compiler-cache')
    import torch
    import torch_npu
    torch.npu._lazy_init = forbidden
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.compiler import CompiledKernel

    class NoRuntime:
        def __getattr__(self, name):
            forbidden()

    triton.runtime.driver.set_active(NoRuntime())
    CompiledKernel._init_handles = forbidden
    spec = importlib.util.spec_from_file_location('offline_expm1_fulltiles', args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = GPUTarget('npu', 'Ascend910B4', 0)
    rows = []
    for dtype, n in [('fp16', 16777216), ('bf16', 16777216),
                     ('fp32', 16777216), ('fp16', 10000002)]:
        row = dict(dtype=dtype, n=n)
        try:
            compiled = triton.compile(ASTSource(module._expm1_streaming,
                dict(a_ptr='*' + dtype), dict(N=n, BLOCK=4096)),
                target=target, options=dict(num_warps=32))
            assert compiled.module is None and compiled.function is None and compiled._run is None
            folder = args.output / (dtype + '-' + str(n))
            folder.mkdir()
            artifacts = {}
            for kind, data in compiled.asm.items():
                if not isinstance(data, (str, bytes)):
                    continue
                path = folder / ('kernel.' + kind)
                path.write_bytes(data.encode() if isinstance(data, str) else data)
                artifacts[kind] = dict(path=str(path), sha256=sha(path))
            row.update(status='compiled_without_runtime', artifacts=artifacts)
        except Exception as error:
            row.update(status='compile_failed', error=type(error).__name__ + ': ' + str(error)[:6000])
        rows.append(row)
        value = dict(source=str(args.source), source_sha256=sha(args.source), checker_sha256=sha(__file__),
            rows=rows, device_executed=False, performance_measured=False,
            scope='显式910B4目标、单CPU、独立缓存；编译成功不代表NPU数值或性能通过。')
        (args.output / 'result.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps({k: v for k, v in row.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
