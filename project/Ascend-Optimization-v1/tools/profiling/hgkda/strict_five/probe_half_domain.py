"""在实际Triton路径检查全部FP16或BF16位模式；仅正确性补测，不产生性能成绩。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='float16')
    args = parser.parse_args()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == '7'
    assert not args.output.exists(), '已有探测身份，不重发未知任务'
    args.output.mkdir(parents=True)
    identity = dict(source=str(args.source), source_sha256=sha(args.source), checker_sha256=sha(__file__))
    (args.output/'started.json').write_text(json.dumps(identity)+'\n', encoding='utf8')
    import torch
    import torch_npu
    torch.set_num_threads(1)
    dtype = getattr(torch, args.dtype)
    spec = importlib.util.spec_from_file_location('half_domain_candidate', args.source)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    base = bits.view(dtype)
    reference_base = torch.expm1(base.to(torch.float64)).to(dtype)
    rows = []
    cases = [('flat-all-bits', 65536, 0), ('streaming-all-bits', 8388608, 0),
             ('streaming-tail-all-bits', 8388611, 0), ('unaligned-all-bits', 8388608, 1)]
    for name, n, offset in cases:
        row = dict(case=name, elements=n, offset=offset, passed=False)
        try:
            cpu = base.repeat((n+65535)//65536)[:n]
            expected = reference_base.repeat((n+65535)//65536)[:n]
            backing = torch.full((n+offset+32,), 37., dtype=dtype, device='npu:0')
            x = backing[offset:offset+n]
            x.copy_(cpu)
            returned = candidate.run(x)
            torch.npu.synchronize()
            actual = x.cpu()
            # CPU和设备的NaN有效载荷不要求相同；数值、非有限类型及零符号独立核验。
            matched = torch.isclose(actual.float(), expected.float(), rtol=.015, atol=0, equal_nan=True)
            zero = cpu == 0
            zero_sign = torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))
            guards = bool((backing[:offset].cpu() == 37).all() and
                          (backing[offset+n:].cpu() == 37).all())
            row.update(matched=int(matched.sum()), total=n, zero_sign_checked=True,
                zero_sign_passed=zero_sign, storage_guards_passed=guards, returned_input_object=returned is x,
                unique_input_bits=65536, rtol=.015, atol=0,
                input_sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())
            failures = (~matched).nonzero().flatten()[:16].tolist()
            row['counterexamples'] = [dict(index=i, input_bits=int(bits[i % 65536]) & 65535,
                input=str(float(cpu[i])), expected=str(float(expected[i])), actual=str(float(actual[i]))) for i in failures]
            row['passed'] = bool(matched.all() and zero_sign and guards and returned is x)
        except Exception as error:
            row['error'] = type(error).__name__+': '+str(error)[:2500]
        rows.append(row)
        value = dict(identity, dtype=args.dtype, status='passed' if len(rows)==4 and all(r['passed'] for r in rows) else 'in_progress',
            passed=sum(r['passed'] for r in rows), total=4, cases=rows,
            performance_measured=False, reference='CPU float64 expm1 then '+args.dtype+' rounding',
            scope=args.dtype+'完整位模式在实际平铺、大输入流式、尾部及非对齐路径的数值补测；不修改正式合同，不计性能，不赋予CANN参考资格。')
        if len(rows)==4:
            value['status'] = 'passed' if all(r['passed'] for r in rows) else 'failed'
        (args.output/'result.json').write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        print(json.dumps({k:row[k] for k in ['case', 'passed']}), flush=True)


if __name__ == '__main__':
    main()
