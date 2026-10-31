"""只读诊断冻结求逆合同：分别保存inverse和info，保留NPU与CPU参考差异。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def describe(value):
    import torch
    value = value.detach().cpu()
    result = dict(shape=list(value.shape), dtype=str(value.dtype), elements=value.numel())
    if value.is_floating_point():
        result.update(nan_count=int(torch.isnan(value).sum()),
                      posinf_count=int(torch.isposinf(value).sum()),
                      neginf_count=int(torch.isneginf(value).sum()))
        finite = value[torch.isfinite(value)]
        if finite.numel():
            result.update(finite_min=float(finite.min()), finite_max=float(finite.max()))
    if value.numel() <= 256:
        result['values'] = value.tolist()
    return result


def summary(probe):
    """模型上下文保留info与非有限分布，大矩阵原值按需读取原件。"""
    def tensor(value):
        return {k: v for k, v in value.items() if k != 'values' or value.get('elements', 0) <= 16}
    cases = []
    for raw in probe['cases']:
        case = {k: raw[k] for k in ['case', 'spec', 'error'] if k in raw}
        case['outputs'] = []
        for output in raw['outputs']:
            entry = {k: output[k] for k in ['output', 'matched']}
            for side in ['npu_reference', 'candidate']:
                entry[side] = tensor(output[side])
            if output.get('error'):
                entry['error'] = output['error'][:700]
            case['outputs'].append(entry)
        cpu = raw.get('cpu_observation', {})
        case['cpu_observation'] = {k: tensor(v) if isinstance(v, dict) else v for k, v in cpu.items()}
        cases.append(case)
    return dict(source=probe['source'], scope=probe['scope'], cases=cases,
                interpretation='大矩阵完整值保留在诊断原件；奇异info和inverse分别判断，不修改冻结合同')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--leaf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('诊断输出已存在，保留原件')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '7':
        raise RuntimeError('仅允许已由控制器分配的物理卡7')
    import torch
    import torch_npu  # noqa: F401
    oracle = imported('inverse_probe_oracle', a.leaf / 'oracle.py')
    candidate = imported('inverse_probe_candidate', a.source)
    definition = json.loads((a.leaf / 'definition.json').read_text(encoding='utf8'))
    names = definition['outputs']
    result = dict(schema='hgkda-inverse-contract-probe-v1', status='completed',
                  source=identity(a.source), oracle=identity(a.leaf / 'oracle.py'),
                  contract={name: identity(a.leaf / name) for name in
                            ['definition.json', 'correctness.jsonl', 'timing.jsonl']},
                  scope='冻结输入的诊断，不改变合同、容差或正式结果；奇异inverse与info分别比较', cases=[])
    cases = [json.loads(line) for line in (a.leaf / 'correctness.jsonl').read_text().splitlines() if line.strip()]
    for case in cases:
        row = dict(case=case['name'], spec=case['inputs']['spec'], tolerance=case['tolerance'], outputs=[])
        result['cases'].append(row)
        try:
            actual_inputs = oracle.gen_inputs(case, 'npu:0')
            expected_inputs = oracle.gen_inputs(case, 'npu:0')
            row['input'] = describe(actual_inputs['A'])
            expected = oracle.run(**expected_inputs)
            actual = candidate.run(**actual_inputs)
            torch.npu.synchronize()
            expected = list(expected) if isinstance(expected, (tuple, list)) else [expected]
            actual = list(actual) if isinstance(actual, (tuple, list)) else [actual]
            row['output_count_matches'] = len(actual) == len(expected) == len(names)
            for name, ref, out in zip(names, expected, actual):
                # 先保存真实返回值，断言失败仍留下info与inverse各自的数据。
                entry = dict(output=name, npu_reference=describe(ref), candidate=describe(out), matched=False)
                row['outputs'].append(entry)
                try:
                    torch.testing.assert_close(out, ref, atol=case['tolerance']['atol'],
                                               rtol=case['tolerance']['rtol'], equal_nan=True)
                    entry['matched'] = True
                except AssertionError as error:
                    entry['error'] = str(error)[:1800]
            try:
                cpu_result = oracle.run(**oracle.gen_inputs(case, 'cpu'))
                cpu_result = list(cpu_result) if isinstance(cpu_result, (tuple, list)) else [cpu_result]
                row['cpu_observation'] = {name: describe(value) for name, value in zip(names, cpu_result)}
            except Exception as error:
                row['cpu_observation'] = dict(status='unavailable', error=type(error).__name__ + ': ' + str(error)[:1000])
        except Exception as error:
            row['error'] = type(error).__name__ + ': ' + str(error)[:2200]
    # 非有限数以字符串保存为标准JSON，便于离线读取原始NaN/Inf图样。
    def standard(value):
        if isinstance(value, float) and not __import__('math').isfinite(value):
            return str(value)
        if isinstance(value, list):
            return [standard(x) for x in value]
        if isinstance(value, dict):
            return {k: standard(v) for k, v in value.items()}
        return value
    result = standard(result)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(dict(status=result['status'], cases=[dict(case=r['case'], error=r.get('error'),
        outputs=[dict(output=x['output'], matched=x['matched']) for x in r['outputs']]) for r in result['cases']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
