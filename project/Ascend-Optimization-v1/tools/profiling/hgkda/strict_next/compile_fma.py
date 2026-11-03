"""单CPU离线编译新FMA提案的三类型平铺及流式路径；不初始化NPU。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    raise RuntimeError('离线编译禁止设备初始化、加载和提交')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--configuration', type=Path,
                        help='可选六实例编译清单；仅允许正常的分块与编译器缓冲选项')
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
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
    spec = importlib.util.spec_from_file_location('strict_next_fma_offline', args.source)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    rows = []
    configurations = []
    for dtype, tiny, block, symbol in [('fp16', .125, 8192, '_expm1_streaming2'),
                                      ('bf16', .125, 8192, '_expm1_streaming1'),
                                      ('fp32', .0078125, 4096, '_expm1_streaming2')]:
        for name, tile in [('_expm1_flat', 4096), (symbol, block)]:
            configurations.append(dict(dtype=dtype, tiny=tiny, symbol=name, block=tile,
                                       options=dict(num_warps=32)))
    if args.configuration:
        configurations = json.loads(args.configuration.read_text(encoding='utf8'))
        assert len(configurations) == 6
        assert all(set(c['options']) <= {'num_warps', 'num_stages', 'multibuffer', 'enable_ubuf_saving'}
                   for c in configurations), '只接受正常编译缓冲选项，不修改流水线或同步'
    for config in configurations:
            dtype, tiny, tile, name = (config[k] for k in ['dtype', 'tiny', 'block', 'symbol'])
            row = dict(dtype=dtype, symbol=name, block=tile, options=config['options'])
            try:
                kernel = triton.compile(ASTSource(getattr(candidate, name),
                    dict(a_ptr='*' + dtype, n='i32'), dict(TINY=tiny, BLOCK=tile)),
                    target=GPUTarget('npu', 'Ascend910B4', 0), options=config['options'])
                assert kernel.module is None and kernel.function is None and kernel._run is None
                out = args.output / (dtype + '-' + name)
                out.mkdir()
                artifacts = {}
                for kind, value in kernel.asm.items():
                    if not isinstance(value, (str, bytes)):
                        continue
                    path = out / ('kernel.' + kind)
                    path.write_bytes(value.encode() if isinstance(value, str) else value)
                    artifacts[kind] = dict(path=str(path), sha256=sha(path))
                row.update(status='compiled_without_runtime', artifacts=artifacts,
                           ttir_fma_count=str(kernel.asm.get('ttir', '')).count('math.fma'))
            except Exception as error:
                row.update(status='compile_failed', error=type(error).__name__ + ': ' + str(error)[:5000])
            rows.append(row)
            result = dict(source_sha256=sha(args.source), checker_sha256=sha(__file__), rows=rows,
                          status='compiled' if len(rows) == 6 and all(r['status'] == 'compiled_without_runtime' for r in rows) else 'in_progress',
                          device_executed=False, performance_measured=False)
            if len(rows) == 6 and result['status'] != 'compiled':
                result['status'] = 'compile_failed'
            (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
            print(json.dumps({k: v for k, v in row.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
