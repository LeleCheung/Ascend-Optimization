"""只读取原始回执，汇总求逆掩码修复闭环；不提交设备任务或重复计时。"""
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
BATCH = HERE.parents[1]
REMOTE = '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/strict-night-20261102/'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = {}

    def load(relative):
        path = BATCH / relative
        evidence[REMOTE + relative] = sha(path)
        return json.loads(path.read_text(encoding='utf8'))

    probe_relative = 'numerical-probes/inverse-mask-reciprocal-v1/'
    probe = load(probe_relative + 'result.json')
    assert probe['status'] == 'measured' and probe['device_executed']
    assert not probe['qualified_CANN'] and not probe['performance_measured']
    modes = [{k: row[k] for k in ['steps', 'mode', 'inactive_changed', 'active_mismatches']}
             for row in probe['modes']]
    assert {(row['steps'], row['mode']) for row in modes} == {
        (steps, mode) for steps in [2, 7] for mode in [0, 1, 2]}
    observed = []
    for steps in [2, 7]:
        inputs = load(probe_relative + f'inputs-{steps}.json')
        total_inactive = sum(inputs['thresholds'])
        rows = {row['mode']: row for row in modes if row['steps'] == steps}
        observed.append(dict(steps=steps, inactive_elements=total_inactive,
            original_changed=rows[0]['inactive_changed'],
            operand_mask_changed=rows[1]['inactive_changed'],
            ordinary_mul_add_changed=rows[2]['inactive_changed']))
        # 这是对已存结果的核对，不是新增数值测试。
        assert rows[0]['inactive_changed'] == rows[2]['inactive_changed'] == total_inactive
        assert rows[1]['inactive_changed'] == 0
        assert all(row['active_mismatches'] == 0 for row in rows.values())

    ir = {}
    for name, tokens in {
        'masked-2-2.ttir': ['scf.for', 'arith.mulf', 'arith.addf', 'arith.select', 'scf.yield'],
        'masked-2-2.ttadapter': ['scf.for', 'tensor.extract_slice', 'tensor.insert_slice'],
        'masked-2-2.bcmlir': ['scf.for', 'tensor.extract_slice', 'tensor.insert_slice'],
    }.items():
        path = BATCH / probe_relative / name
        digest = sha(path)
        evidence[REMOTE + probe_relative + name] = digest
        lines = path.read_text(encoding='utf8').splitlines()
        assert all(any(token in line for line in lines) for token in tokens)
        ir[name] = dict(sha256=digest, lines=[dict(line=i, text=line.strip())
            for i, line in enumerate(lines, 1)
            if not line.startswith('#loc') and any(token in line for token in tokens)])

    versions = {}
    for name in ['codex-column-solve-v6', 'codex-masked-operands-v7']:
        prefix = 'proposals/linalg_inv_ex/' + name + '/'
        source = BATCH / prefix / 'main.py'
        evidence[REMOTE + prefix + 'main.py'] = sha(source)
        result = load(prefix + 'pytorch/Codex-strict-PyTorch-1.result.json')
        versions[name] = dict(source_sha256=sha(source),
            formal={k: result[k] for k in ['status', 'num_passed', 'num_workloads', 'geo_mean']})
        if name.endswith('v7'):
            extra = load(prefix + 'extra-pytorch.json')
            backend = load(prefix + 'backend-verification.json')
            assert extra['status'] == 'passed' and extra['source_sha256'] == sha(source)
            assert backend['status'] == 'verified' and backend['source_sha256'] == sha(source)
            assert result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads'] == 7
            versions[name].update(supplement={k: extra[k] for k in ['status', 'passed', 'total']},
                                 backend='verified')
    reciprocals = [{k: row[k] for k in ['denominator', 'old_mismatches', 'vector_mismatches']}
                   for row in probe['reciprocal']]
    assert sum(row['vector_mismatches'] for row in reciprocals) == 0
    value = dict(schema='inverse-mask-closure-v1', status='measured_workaround_verified',
        mechanisms=observed, reciprocal=reciprocals, ir=ir, versions=versions,
        first_counterexample=probe['modes'][0]['first_inactive'][0],
        conclusions=[
            '原where更新及普通乘加where更新在两种循环次数下均改写全部34个非活跃元素；先屏蔽操作数的版本没有此差异。',
            '原普通乘加版本的TTIR仍保留select，adapter将其转换为切片保留；尚未定位造成错误的具体后端pass或ISA。',
            '求逆v7将操作数掩码与非splat分母两项修改一起应用，正式由v6的3/7变为7/7，补测8/8并完成实际Triton后端核验。',
            '两个机制各有最小探测，但完整v7没有逐项消融，不能把7/7的贡献拆成独立比例。',
            '77.743倍是相对当前含CPU回退的原PyTorch完整调用；没有合格CANN完整参考，不增加25/30验收数。'],
        scope='仅汇总已执行的有限输入和保存产物，不是全IEEE或全部编译版本的证明；本脚本无设备执行与性能计时。',
        evidence_sha256=evidence, analyzer_sha256=sha(Path(__file__)))
    target = HERE / 'mask-closure-v1.json'
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps(dict(status=value['status'], mechanisms=observed,
                         v7=versions['codex-masked-operands-v7'], evidence_files=len(evidence)), ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()
