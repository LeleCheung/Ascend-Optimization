"""按冻结输入比较求逆原稿和阶段输出，并检查标量/向量混合算术；不计时。"""
import argparse
import hashlib
import json
from pathlib import Path

PROBE_NAME = 'inverse-stages-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process(runner):
    from night_common import PY, read
    out = runner.batch / 'numerical-probes' / PROBE_NAME
    spec = out / 'spec.json'
    if not spec.exists():
        return
    configuration = read(spec)
    assert configuration['checker_sha256'] == sha(__file__)
    assert configuration['source_sha256'] == sha(configuration['source'])
    result = out / 'result.json'
    key = 'diagnostic:' + PROBE_NAME
    previous = runner.state['jobs'].get(key, {})
    if not result.exists():
        assert previous.get('status') not in {'running', 'completed'}, '未知回执不重发'
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
    assert value['source_sha256'] == configuration['source_sha256']
    assert value['qualified_CANN'] is False and value['performance_measured'] is False
    runner.state['jobs'][key] = dict(status='completed', result=str(result),
                                   result_sha256=sha(result), outcome=value['status'])
    runner.persist()


def instrument(text):
    start = text.index('@triton.jit\ndef _leftlooking')
    end = text.index('@triton.jit\ndef _large')
    kernel = text[start:end]
    kernel = kernel.replace('def _leftlooking(A, INV, INFO,', 'def _leftlooking(A, INV, INFO, DUMP,')
    kernel = kernel.replace('TRANSPOSE: tl.constexpr):', 'TRANSPOSE: tl.constexpr, COUNT: tl.constexpr):')
    address = 'pid * B * B + r[:, None] * B + c[None, :]'

    def dump(stage, value):
        return 'tl.store(DUMP + ' + str(stage) + ' * COUNT * B * B + ' + address + ', ' + value + ')\n'

    marker = '    rhs = tl.where(r[:, None] == c[None, :], 1., 0.)'
    assert kernel.count(marker) == 1
    kernel = kernel.replace(marker, '    ' + dump(0, 'work') + marker)
    marker = '        for back in tl.range(0, N):\n            j = N - 1 - back\n            lcol ='
    assert kernel.count(marker) == 1
    kernel = kernel.replace(marker, '        ' + dump(1, 'rhs') + marker)
    marker = '        for back in tl.range(0, N):\n            j = N - 1 - back\n            p ='
    assert kernel.count(marker) == 1
    kernel = kernel.replace(marker, '        ' + dump(2, 'rhs') + marker)
    return text[:start] + kernel + text[end:]


SCALAR_KERNEL = '''
@triton.jit
def _scalar_math(A, B, C, F, D, R, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i)
    b = tl.load(B + pid)
    c = tl.load(C + i)
    tl.store(F + i, _compensated_fma(a, b, c))
    tl.store(D + i, _corrected_div(a, b))
    tl.store(R + i, _corrected_div(tl.full((BLOCK,), 1., tl.float32), b))
'''


def measure(configuration, out):
    import ctypes
    import ctypes.util
    import importlib.util
    import struct
    import torch
    import torch_npu
    import triton

    source = Path(configuration['source'])
    leaf = Path(configuration['leaf'])
    assert sha(source) == configuration['source_sha256']

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    original = load('inverse_stages_original', source)
    variant = out / 'instrumented.py'
    assert not variant.exists()
    variant.write_text(instrument(source.read_text(encoding='utf8')) + SCALAR_KERNEL, encoding='utf8')
    modified = load('inverse_stages_instrumented', variant)
    oracle = load('inverse_stages_oracle', leaf / 'oracle.py')
    rows = []
    for line in (leaf / 'correctness.jsonl').read_text(encoding='utf8').splitlines():
        context = json.loads(line)
        inputs = oracle.gen_inputs(context, 'npu:0')
        a = inputs['A']
        assert a.is_contiguous(), '本诊断只解释冻结的连续输入分支'
        n = a.shape[-1]
        block = max(16, triton.next_power_of_2(n))
        count = a.numel() // (n * n)
        original_value = original.run(**inputs)
        expected = oracle.run(**inputs)
        original_record = dict(case=context['name'], input=a.cpu().tolist(),
                               original=original_value.inverse.cpu().tolist(),
                               original_info=original_value.info.cpu().tolist(),
                               reference=expected.inverse.cpu().tolist(),
                               reference_info=expected.info.cpu().tolist())
        (out / (context['name'] + '-original.json')).write_text(
            json.dumps(original_record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        inverse = torch.empty_like(a)
        info = torch.empty(tuple(a.shape[:-2]), dtype=torch.int32, device=a.device)
        dump = torch.empty((3, count, block, block), dtype=torch.float32, device=a.device)
        compiled = modified._leftlooking[(count,)](
            a, inverse, info, dump, n, block, tuple(a.shape[:-2]), tuple(a.stride()), True, count,
            enable_fp_fusion=False, num_warps=32)
        torch.npu.synchronize()
        cpu = a.cpu()
        lu, pivots, lu_info = torch.linalg.lu_factor_ex(cpu.mT())
        row = dict(case=context['name'], input=cpu.tolist(), reference=expected.inverse.cpu().tolist(),
                   reference_info=expected.info.cpu().tolist(), original=original_value.inverse.cpu().tolist(),
                   original_info=original_value.info.cpu().tolist(), instrumented=inverse.cpu().tolist(),
                   instrumented_info=info.cpu().tolist(), stages=dump.cpu().tolist(),
                   cpu_lu=lu.tolist(), cpu_pivots=pivots.tolist(), cpu_lu_info=lu_info.tolist())
        row['instrumentation_matches_original'] = bool(torch.equal(inverse.cpu(), original_value.inverse.cpu()))
        artifacts = {}
        for kind, data in compiled.asm.items():
            if not isinstance(data, (str, bytes)):
                continue
            path = out / (context['name'] + '.' + kind)
            path.write_bytes(data.encode() if isinstance(data, str) else data)
            artifacts[kind] = dict(path=str(path), sha256=sha(path))
        row['artifacts'] = artifacts
        (out / (context['name'] + '.json')).write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        rows.append(row)
    generator = torch.Generator().manual_seed(715037)
    block = 128
    denominators = torch.tensor([1.234567, 4.410436, 3.49273, -.173, 1.e-8, -3.e-10, .5, 9.1], dtype=torch.float32)
    a = torch.randn(8 * block, generator=generator)
    c = -(a * denominators.repeat_interleave(block))
    outputs = [torch.empty_like(a, device='npu:0') for _ in range(3)]
    modified._scalar_math[(8,)](a.to('npu:0'), denominators.to('npu:0'), c.to('npu:0'), *outputs, block,
                                enable_fp_fusion=False, num_warps=32)
    torch.npu.synchronize()
    values = [v.cpu().tolist() for v in outputs]
    libm = ctypes.CDLL(ctypes.util.find_library('m'))
    fmaf = libm.fmaf
    fmaf.argtypes = [ctypes.c_float] * 3
    fmaf.restype = ctypes.c_float
    f32 = lambda value: struct.unpack('<f', struct.pack('<f', value))[0]
    bits = lambda value: struct.pack('<f', value)
    mismatches = dict.fromkeys(['compensated_fma', 'corrected_div', 'corrected_reciprocal'], 0)
    scalar_rows = []
    for i, (x, z) in enumerate(zip(a.tolist(), c.tolist())):
        y = denominators[i // block].item()
        expected = [fmaf(x, y, z), f32(x / y), f32(1. / y)]
        row = dict(index=i, a=x, b=y, c=z)
        for name, data, target in zip(mismatches, values, expected):
            row[name] = data[i]
            row['expected_' + name] = target
            mismatches[name] += bits(data[i]) != bits(target)
        scalar_rows.append(row)
    return dict(status='measured', cases=rows, scalar_operand_mismatches=mismatches,
                scalar_rows=scalar_rows, variant_sha256=sha(variant), device_executed=True,
                scope='只诊断冻结输入的原稿、带中间存储的变体和标量/向量混合算术；插桩可能改变编译行为，先核对两者输出。不是性能评测或CANN参考资格。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    configuration = json.loads(args.spec.read_text(encoding='utf8'))
    assert configuration['checker_sha256'] == sha(__file__)
    try:
        value = measure(configuration, args.output.parent)
    except Exception as error:
        value = dict(status='diagnostic_error', error=type(error).__name__ + ': ' + str(error)[:5000])
    value.update(checker_sha256=sha(__file__), spec_sha256=sha(args.spec),
                 source_sha256=configuration['source_sha256'], qualified_CANN=False, performance_measured=False)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({key: value[key] for key in ['status', 'scalar_operand_mismatches', 'error'] if key in value}))


if __name__ == '__main__':
    main()
