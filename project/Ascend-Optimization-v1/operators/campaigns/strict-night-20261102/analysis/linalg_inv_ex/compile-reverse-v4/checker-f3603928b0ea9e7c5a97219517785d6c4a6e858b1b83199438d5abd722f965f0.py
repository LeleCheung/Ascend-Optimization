"""单CPU离线编译求逆候选；显式指定目标，阻止设备初始化和kernel提交。"""
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
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TRITON_ASCEND_ARCH'] = 'Ascend910B4'
    # 隔离编译缓存；不改当前KG、其他用户或NPU评测的缓存。
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
    spec = importlib.util.spec_from_file_location('hgkda_compile_inverse_only', args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = GPUTarget('npu', 'Ascend910B4', 0)
    rows = []
    cases = [('_leftlooking', 3, True), ('_leftlooking', 7, True),
             ('_leftlooking', 7, False), ('_large', 64, True)]
    for name, n, transpose in cases:
        constants = dict(N=n, B=max(16, triton.next_power_of_2(n)), TRANSPOSE=transpose)
        signature = dict(A='*fp32', INV='*fp32', INFO='*i32')
        if name == '_leftlooking':
            batch = (2,) if n == 7 else ()
            strides = (n * n, n, 1) if batch else (n, 1)
            constants.update(BSHAPE=batch, STRIDES=strides if transpose else tuple(s * 2 for s in strides))
        else:
            constants.update(CT=32)
            signature['PIVOTS'] = '*i32'
        row = dict(kernel=name, n=n, transpose=transpose, constexprs=constants)
        try:
            compiled = triton.compile(ASTSource(getattr(module, name), signature, constants),
                                      target=target, options=dict(num_warps=4 if n <= 8 else 8,
                                                                  enable_fp_fusion=n > 8))
            assert compiled.module is None and compiled.function is None and compiled._run is None
            out = args.output / (name.lstrip('_') + '-' + str(n) + '-' + str(transpose).lower())
            out.mkdir()
            artifacts = {}
            for kind, data in compiled.asm.items():
                if not isinstance(data, (str, bytes)):
                    continue
                path = out / ('kernel.' + kind)
                path.write_bytes(data.encode() if isinstance(data, str) else data)
                artifacts[kind] = dict(path=str(path), sha256=sha(path))
            row.update(status='compiled_without_runtime', artifacts=artifacts)
        except Exception as error:
            row.update(status='compile_failed', error=type(error).__name__ + ': ' + str(error)[:6000])
        rows.append(row)
        (args.output / 'result.json').write_text(json.dumps(dict(
            source=str(args.source), source_sha256=sha(args.source), checker_sha256=sha(__file__),
            rows=rows, device_executed=False, performance_measured=False,
            scope='单CPU显式目标交叉编译；阻止driver/设备初始化及CompiledKernel加载。编译成功不证明数值或性能。'),
            ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps({k: v for k, v in row.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
