"""对直接舍入反例验证纯Triton补偿乘加／商修正；只作数值诊断。"""
import argparse
import hashlib
import json
from pathlib import Path


TRITON_HELPERS = '''@triton.jit
def _compensated_fma(a, b, c):
    product = a * b
    split = product + c
    mask = tl.full((), 0xfffff000, tl.uint32)
    ah = (a.to(tl.uint32, bitcast=True) & mask).to(tl.float32, bitcast=True)
    bh = (b.to(tl.uint32, bitcast=True) & mask).to(tl.float32, bitcast=True)
    al = a - ah
    bl = b - bh
    e1 = product - ah * bh
    e2 = e1 - al * bh
    e3 = e2 - ah * bl
    product_error = al * bl - e3
    virtual = split - product
    sum_error = (product - (split - virtual)) + (c - virtual)
    return split + (sum_error + product_error)


@triton.jit
def _corrected_div(a, b):
    quotient = tl.div_rn(a, b)
    residual = _compensated_fma(-quotient, b, a)
    correction = tl.div_rn(residual, b)
    corrected = quotient + correction
    ordinary = (b == 0.) | (tl.abs(quotient) == float('inf')) | (quotient != quotient)
    return tl.where(ordinary, quotient, corrected)


'''

KERNEL = '''import triton
import triton.language as tl

''' + TRITON_HELPERS + '''@triton.jit
def arithmetic(A, B, C, F, S, H, D, Q, R, RC, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    a = tl.load(A + i, i < N, other=1.)
    b = tl.load(B + i, i < N, other=1.)
    c = tl.load(C + i, i < N, other=0.)
    tl.store(F + i, tl.fma(a, b, c), i < N)
    tl.store(S + i, a * b + c, i < N)
    tl.store(H + i, _compensated_fma(a, b, c), i < N)
    tl.store(D + i, tl.div_rn(a, b), i < N)
    tl.store(Q + i, _corrected_div(a, b), i < N)
    tl.store(R + i, tl.div_rn(1., b), i < N)
    tl.store(RC + i, _corrected_div(1., b), i < N)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process(runner):
    from night_common import PY, read, save
    out = runner.batch / 'numerical-probes/fp32-compensated-v2'
    spec = out / 'spec.json'
    if not spec.exists():
        return
    configuration = read(spec)
    assert configuration['checker_sha256'] == sha(__file__)
    result = out / 'result.json'
    key = 'diagnostic:fp32-compensated-v2'
    previous = runner.state['jobs'].get(key, {})
    if not result.exists():
        assert previous.get('status') not in {'running', 'completed'}, '未知回执不重新提交'
        frozen = out / ('checker-' + sha(__file__) + '.py')
        if frozen.exists():
            assert frozen.read_bytes() == Path(__file__).read_bytes()
        else:
            frozen.write_bytes(Path(__file__).read_bytes())
        original = runner.state.get('current_job')
        runner.state['current_job'] = key
        runner.state['jobs'][key] = dict(status='running', checker=str(frozen),
                                       spec=str(spec), spec_sha256=sha(spec))
        runner.persist()
        runner.cmd(key, [PY, frozen, '--spec', spec, '--output', result], 900)
        runner.state['current_job'] = original
    value = read(result)
    assert value['checker_sha256'] == configuration['checker_sha256']
    assert value['spec_sha256'] == sha(spec)
    assert value['qualified_CANN'] is False and value['performance_measured'] is False
    runner.state['jobs'][key] = dict(status='completed', result=str(result),
                                   result_sha256=sha(result), outcome=value['status'])
    runner.persist()


def measure(out):
    import ctypes
    import ctypes.util
    import importlib.util
    import struct
    import torch
    import torch_npu
    import triton

    kernel_path = out.parent / 'kernel.py'
    assert not kernel_path.exists()
    kernel_path.write_text(KERNEL, encoding='utf8')
    spec = importlib.util.spec_from_file_location('hgkda_compensated_probe', kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    count = 8192
    generator = torch.Generator().manual_seed(937102)
    a = torch.randn(count, generator=generator, dtype=torch.float32)
    b = torch.randn(count, generator=generator, dtype=torch.float32)
    b = torch.where(b.abs() < .01, torch.full_like(b, .01), b)
    c = torch.randn(count, generator=generator, dtype=torch.float32)
    # 普通指数范围；覆盖消去，不将此有限样本说成全IEEE证明。
    exponents = torch.randint(-20, 21, (3, count), generator=generator)
    a = torch.ldexp(a, exponents[0])
    b = torch.ldexp(b, exponents[1])
    c = torch.ldexp(c, exponents[2])
    c[:count // 2] = -(a[:count // 2] * b[:count // 2])
    a[0] = 1. + 2. ** -23
    b[0] = 1. - 2. ** -23
    c[0] = -1.
    cpu = [x.tolist() for x in [a, b, c]]
    raw = struct.pack('<' + str(3 * count) + 'f', *(cpu[0] + cpu[1] + cpu[2]))
    inputs = out.parent / 'input-fp32.bin'
    assert not inputs.exists()
    inputs.write_bytes(raw)
    device = [x.to('npu:0') for x in [a, b, c]]
    libm = ctypes.CDLL(ctypes.util.find_library('m'))
    fmaf = libm.fmaf
    fmaf.argtypes = [ctypes.c_float] * 3
    fmaf.restype = ctypes.c_float

    def bits(x):
        return struct.pack('<f', x)

    def f32(x):
        return struct.unpack('<f', bits(x))[0]

    batches = []
    for fusion in [False, True]:
        outputs = [torch.empty((count,), dtype=torch.float32, device='npu:0') for _ in range(7)]
        compiled = module.arithmetic[(triton.cdiv(count, 256),)](
            *device, *outputs, count, enable_fp_fusion=fusion)
        torch.npu.synchronize()
        values = [x.cpu().tolist() for x in outputs]
        names = ['fma', 'split', 'compensated', 'div_rn', 'corrected_div', 'reciprocal', 'corrected_reciprocal']
        errors = dict.fromkeys(names, 0)
        rows = []
        for i, (x, y, z) in enumerate(zip(*cpu)):
            targets = [fmaf(x, y, z)] * 3 + [f32(x / y)] * 2 + [f32(1. / y)] * 2
            row = dict(index=i, a=x, b=y, c=z, expected_fma=targets[0],
                       expected_div=targets[3], expected_reciprocal=targets[5])
            for name, data, expected in zip(names, values, targets):
                row[name] = data[i]
                row[name + '_exact'] = bits(data[i]) == bits(expected)
                errors[name] += not row[name + '_exact']
            rows.append(row)
        artifacts = {}
        for name, data in compiled.asm.items():
            if isinstance(data, (str, bytes)):
                path = out.parent / ('fusion-' + str(fusion).lower() + '.' + name)
                path.write_bytes(data.encode() if isinstance(data, str) else data)
                artifacts[name] = dict(path=str(path), sha256=sha(path))
        batches.append(dict(enable_fp_fusion=fusion, exact_mismatches=errors,
                            rows=rows, artifacts=artifacts))
    return dict(status='measured', samples=count, batches=batches,
                input=str(inputs), input_sha256=sha(inputs),
                kernel=str(kernel_path), kernel_sha256=sha(kernel_path),
                torch=torch.__version__, torch_npu=torch_npu.__version__,
                device=str(torch.npu.get_device_name(0)),
                scope='普通指数范围随机值及乘积消去；CPU libm fmaf和FP32商为参考。有限样本数值诊断，无性能或CANN资格结论。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), '已有结果不覆盖或重发'
    configuration = json.loads(args.spec.read_text(encoding='utf8'))
    assert configuration['checker_sha256'] == sha(__file__)
    try:
        value = measure(args.output)
    except Exception as error:
        value = dict(status='diagnostic_error', error=type(error).__name__ + ': ' + str(error)[:4000])
    value.update(checker_sha256=sha(__file__), spec_sha256=sha(args.spec),
                 qualified_CANN=False, performance_measured=False)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(status=value['status'], error=value.get('error'),
                         batches=[{k: b[k] for k in ['enable_fp_fusion', 'exact_mismatches']}
                                  for b in value.get('batches', [])])))


if __name__ == '__main__':
    main()
