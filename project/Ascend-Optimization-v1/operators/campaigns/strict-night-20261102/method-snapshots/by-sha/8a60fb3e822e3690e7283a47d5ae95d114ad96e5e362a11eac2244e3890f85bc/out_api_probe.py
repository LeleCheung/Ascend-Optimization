"""补验舍入算子的可选out与输出尺寸；原KGS输入和容差保持不变。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

MODES = {'omitted', 'explicit_none', 'empty_out', 'reshape_out'}
DTYPES = {'float16', 'bfloat16', 'float32'}


def qualified(record, source_sha256):
    required = {(dtype, mode) for dtype in DTYPES for mode in MODES}
    cases = record.get('cases', [])
    return (record.get('status') == 'passed' and record.get('source_sha256') == source_sha256
            and len(cases) == len(required)
            and {(c.get('dtype'), c.get('mode')) for c in cases} == required
            and all(c.get('passed') is True for c in cases))


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator', choices=['round_out', 'special_round_out'], required=True)
    for name in ['source', 'leaf', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), '已有结果不覆盖或重复提交'
    import torch
    import torch_npu  # noqa: F401
    candidate = imported('out_api_candidate', args.source)
    oracle = imported('out_api_original_oracle', args.leaf/'oracle.py')
    source_sha256 = hashlib.sha256(args.source.read_bytes()).hexdigest()
    cases = [json.loads(line) for line in (args.leaf/'correctness.jsonl').read_text().splitlines() if line.strip()]
    rows = []
    for dtype in sorted(DTYPES):
        case = next(c for c in cases if c['inputs']['spec']['dtype'] == dtype
                    and all(n > 0 for n in c['inputs']['spec']['shape']))
        for mode in sorted(MODES):
            row = dict(dtype=dtype, mode=mode, original_case=case['name'], passed=False)
            try:
                inputs = oracle.gen_inputs(case, 'npu:0')
                ref_inputs = oracle.gen_inputs(case, 'npu:0')
                name = 'input' if args.operator == 'round_out' else 'self'
                original = inputs[name].clone()
                for payload in [inputs, ref_inputs]:
                    tensor = payload[name]
                    if mode == 'omitted':
                        payload.pop('out')
                    elif mode == 'explicit_none':
                        payload['out'] = None
                    elif mode == 'empty_out':
                        payload['out'] = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
                    else:
                        payload['out'] = torch.empty((1, tensor.numel()), dtype=tensor.dtype, device=tensor.device)
                target = inputs.get('out')
                expected = oracle.run(**ref_inputs)
                actual = candidate.run(**inputs)
                torch.npu.synchronize()
                tolerance = case['tolerance']
                torch.testing.assert_close(actual, expected, atol=tolerance['atol'],
                                           rtol=tolerance['rtol'], equal_nan=True)
                torch.testing.assert_close(inputs[name], original, atol=0, rtol=0, equal_nan=True)
                if target is not None:
                    assert actual is target, '未返回调用者out对象'
                else:
                    assert not torch._C._is_alias_of(actual, inputs[name]), '默认out复用了输入'
                assert actual.shape == inputs[name].shape
                row['passed'] = True
            except Exception as error:
                row['error'] = type(error).__name__+': '+str(error)[:2000]
            rows.append(row)
    record = dict(status='passed' if all(r['passed'] for r in rows) else 'failed',
                  operator=args.operator, source_sha256=source_sha256,
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  oracle_sha256=hashlib.sha256((args.leaf/'oracle.py').read_bytes()).hexdigest(),
                  cases=rows, scope='原PyTorch语义下12项可选out/尺寸补测；不修改正式合同，不参与性能计时')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=record['status'], passed=sum(r['passed'] for r in rows), total=len(rows))))


if __name__ == '__main__':
    main()
