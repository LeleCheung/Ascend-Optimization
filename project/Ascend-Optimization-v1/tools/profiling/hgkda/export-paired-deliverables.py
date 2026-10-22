#!/usr/bin/env python3
"""验证本批单次完整评测与语义身份，按真实延迟选择交付源码。"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geo(values):
    return math.exp(sum(math.log(x) for x in values)/len(values))


def read_arm(batch, op, arm, require_dense=False):
    directory = batch/'results'/op/arm
    semantics_path = directory/'semantics.json'
    source = (batch/'candidates'/(op+'-'+arm.removeprefix('codex-')+'.py') if arm.startswith('codex-')
              else directory/'export/main.py')
    if not semantics_path.exists() or not source.exists():
        return None
    semantics = json.loads(semantics_path.read_text())
    if semantics.get('status') != 'passed':
        return None
    if (semantics.get('source_sha256') != sha(source) or semantics.get('operator') != op
            or not semantics.get('passed') == semantics.get('total') == 42
            or semantics.get('physical_device') != 7 or semantics.get('logical_device') != 'npu:0'
            or semantics.get('checker_sha256') != sha(Path(__file__).with_name('check-pointwise-semantics.py'))):
        raise ValueError('语义检查身份或数量不符：'+str(directory))
    rows = semantics['cases']
    expected = {(dtype, case) for dtype in ['torch.float16','torch.bfloat16','torch.float32'] for case in range(14)}
    if (len(rows) != 42 or {(r['dtype'],r['case']) for r in rows} != expected
            or any(r['status'] != 'passed' or r.get('repetitions') != 2 for r in rows)):
        raise ValueError('语义case集合或重复执行不完整')
    dense_path = directory/'dense-semantics.json'
    dense_count = 0
    if require_dense:
        if not dense_path.is_file():
            raise ValueError('最终交付缺少连续对齐快路径检查：'+str(directory))
        dense = json.loads(dense_path.read_text())
        if dense.get('status') == 'failed':
            return None
        expected_dense = {(dtype, shape) for dtype in ['torch.float16','torch.bfloat16','torch.float32']
                          for shape in [(1024,), (65536,), (1048576,), (4096,4096)]}
        dense_rows = dense.get('cases', [])
        if (dense.get('status') != 'passed' or dense.get('source_sha256') != sha(source)
                or dense.get('operator') != op or dense.get('passed') != 12 or dense.get('total') != 12
                or dense.get('physical_device') != 7 or dense.get('logical_device') != 'npu:0'
                or dense.get('checker_sha256') != sha(Path(__file__).with_name('check-dense-pointwise.py'))
                or len(dense_rows) != 12
                or {(r['dtype'], tuple(r['shape'])) for r in dense_rows} != expected_dense
                or any(r['status'] != 'passed' or r.get('repetitions') != 2
                       or not r.get('stream_switch_checked') for r in dense_rows)):
            raise ValueError('连续对齐检查身份或case覆盖不完整：'+str(directory))
        dense_count = len(dense_rows)
    formal = directory/'formal'
    results = list(formal.glob('*.result.json'))
    if len(results) != 1:
        raise ValueError('本批要求一次正式完整结果：'+str(formal))
    result_path = results[0]
    stem = result_path.name[:-len('.result.json')]
    request_path = formal/(stem+'.request.json')
    request, result = [json.loads(p.read_text()) for p in [request_path, result_path]]
    submitted = request['implementation']['sources']
    if len(submitted) != 1 or submitted[0]['content'] != source.read_text():
        raise ValueError('实际提交内容与交付源码不同')
    contracts = batch/'catalog/ops'/op
    workloads = [json.loads(line)['name'] for name in ['correctness.jsonl','timing.jsonl']
                 for line in (contracts/name).read_text().splitlines() if line.strip()]
    phases = {json.loads(line)['name']: phase for phase in ['correctness','timing']
              for line in (contracts/(phase+'.jsonl')).read_text().splitlines() if line.strip()}
    actual = result.get('per_workload', [])
    if (result.get('status') != 'PASSED' or result.get('num_passed') != len(workloads)
            or result.get('num_workloads') != len(workloads) or len(actual) != len(workloads)
            or {r['uuid'] for r in actual} != set(workloads)
            or any(r['status'] != 'PASSED' or r.get('phase') != phases[r['uuid']] for r in actual)):
        raise ValueError('正式结果未完整覆盖合同')
    timing = [r for r in actual if r.get('phase') == 'timing']
    score = geo([r['reference_latency_ms']/r['latency_ms'] for r in timing])
    if not math.isclose(score, result['geo_mean'], rel_tol=1e-8):
        raise ValueError('逐case重算与汇总加速比不符')
    return dict(operator=op, arm=arm, source=str(source), source_sha256=sha(source),
        submitted_source_sha256=hashlib.sha256(submitted[0]['content'].encode()).hexdigest(),
        newline_policy='原始源码字节与提交文本分别计算SHA；read_text只规范化换行',
        geo_mean=score, geometric_latency_ms=geo([r['latency_ms'] for r in timing]),
        timing_cases=len(timing), workloads=len(actual), semantics_cases=len(rows), dense_cases=dense_count,
        evidence=dict(semantics=str(semantics_path), dense_semantics=str(dense_path) if require_dense else None,
                      request=str(request_path), result=str(result_path)),
        passed=True, meets_08=score >= .8)


def export(batch, output, arms=None, require_dense=False):
    if output.exists():
        raise ValueError('交付目录已存在，请保留原始版本')
    output.mkdir(parents=True)
    records = []
    plan = json.loads((batch/'plan.json').read_text())
    for record in plan['records']:
        for name, digest in record['contract_sha256'].items():
            if sha(batch/record['catalog_leaf']/name) != digest:
                raise ValueError('冻结合同已改变')
    for op in ['log10_', 'selu_']:
        alternatives = [v for arm in (arms or ['codex-rawptr-v1','paired-HG-KDA'])
                        if (v := read_arm(batch, op, arm, require_dense=require_dense)) is not None]
        if not alternatives:
            raise ValueError('没有可交付版本：'+op)
        # 使用同case候选的实际延迟排序，避免参考偶发变慢改变候选选择。
        best = min(alternatives, key=lambda x:x['geometric_latency_ms'])
        target = output/op/best['source_sha256']
        target.mkdir(parents=True)
        (target/'main.py').write_bytes(Path(best['source']).read_bytes())
        records.append(dict(operator=op, selected=best, alternatives=alternatives,
            delivered=str(target/'main.py'), selection='同合同timing case候选延迟几何平均最小'))
    result = dict(schema_version='hgkda-paired-delivery-v2' if require_dense else 'hgkda-paired-delivery-v1', batch=batch.name,
        records=records, timing_scope='walltime', final_repeats=1,
        dense_gate_required=require_dense,
        all_meet_08=all(r['selected']['meets_08'] for r in records))
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# 本批交付结果', '', '均为KGS原36项完整调用计时，相对PyTorch；最终版本一次完整评测。', '',
        '| 算子 | 交付来源 | 相对PyTorch | 正式测试 | 扩展语义 | 连续快路径 |', '|---|---|---:|---:|---:|---:|']
    for r in records:
        b = r['selected']
        lines.append('| `%s` | %s | %.3f× | %d/%d | %d/%d | %s |' % (
            r['operator'], b['arm'], b['geo_mean'], b['workloads'], b['workloads'],
            b['semantics_cases'],b['semantics_cases'],
            '%d/%d' % (b['dense_cases'], b['dense_cases']) if require_dense else '本次未作门禁'))
    lines += ['', 'Codex诊断修复与CC追加生成分别列在manifest中；不将两者混称纯原生KG独立成果。',
        '完整调用、设备OpProf与独立微探针分别解释；数值Roofline尚未建立。']
    (output/'结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arms', nargs='+')
    p.add_argument('--require-dense', action='store_true')
    a = p.parse_args()
    print(json.dumps(export(a.batch_dir, a.output, a.arms, a.require_dense), ensure_ascii=False, indent=2))
