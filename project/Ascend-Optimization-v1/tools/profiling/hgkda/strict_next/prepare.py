"""冻结25项交付和五项最新起点，准备独立续跑批次；不使用设备。"""
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[4]
PREVIOUS = PROJECT / 'operators/campaigns/strict-five-20261103'
BATCH = PROJECT / 'operators/campaigns/strict-next-20261103'
REMOTE = '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/'
OPERATORS = ['cumulative_trapezoid', 'expm1_', 'uniform', 'linalg_inv_ex', 'fractional_max_pool3d']
FILES = ['definition.json', 'oracle.py', 'correctness.jsonl', 'timing.jsonl']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')


def main():
    assert not BATCH.exists(), '新批次已存在；读取实际状态，不重新执行准备入口'
    previous = json.loads((PREVIOUS / 'delivery.json').read_text(encoding='utf8'))
    audit = json.loads((PREVIOUS / 'final-audit.json').read_text(encoding='utf8'))
    assert previous['accepted'] == 25 and len(previous['rows']) == 30
    assert audit['status'] == 'verified' and audit['delivery_sha256'] == sha(PREVIOUS / 'delivery.json')
    BATCH.mkdir(parents=True)
    shutil.copyfile(PREVIOUS / 'delivery.json', BATCH / 'starting-delivery.json')
    records = []
    for op in OPERATORS:
        row = next(v for v in previous['rows'] if v['operator'] == op)
        assert not row['accepted'] and row['source'].startswith(REMOTE)
        source = PROJECT / row['source'][len(REMOTE):]
        assert sha(source) == row['source_sha256']
        seed = BATCH / 'starting-points' / (op + '.py')
        seed.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, seed)
        leaf = BATCH / 'catalog/ops' / op
        shutil.copytree(PREVIOUS / 'catalog/ops' / op, leaf, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(leaf, BATCH / 'original-catalog/ops' / op)
        records.append(dict(operator=op, seed=seed.relative_to(BATCH).as_posix(), source_sha256=sha(seed),
                            catalog_leaf=leaf.relative_to(BATCH).as_posix(), reference='original_PyTorch_only',
                            contract_sha256={name: sha(leaf / name) for name in FILES}))
    for name in ['catalog', 'original-catalog']:
        save(BATCH / name / 'manifest.json', dict(api_version='v6.2', name=BATCH.name + '-' + name,
                                                evaluator='native', layout='per-operator'))
    plan = dict(schema='hgkda-strict-next-v1', starting_commit='9798acae1accdc9078ca68fdbf918096ebce14e5',
                starting_delivery_sha256=sha(PREVIOUS / 'delivery.json'), frozen_accepted=25, target=30,
                server='http://127.0.0.1:19659', physical_card=7, records=records,
                timing_scope='walltime', passes=[dict(rounds=5, budget_seconds=3600)],
                candidate_backend='strict_Triton', reference_policy='合格原生CANN／来源明确Ascend C；原PyTorch单列',
                measurement_policy='原合同、容差、输入和默认确定性不变；每份新源码一次完整评测，复用既有缓存，不重发未知任务')
    save(BATCH / 'plan.json', plan)
    leaf = BATCH / 'catalog/ops/expm1_'
    cases = [json.loads(line) for line in (leaf / 'timing.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    source = BATCH / 'starting-points/expm1_.py'
    for case in cases:
        spec = case['inputs']['spec']
        if spec['shape'] != [4096, 4096]:
            continue
        save(BATCH / 'analysis/expm1_' / (spec['dtype'] + '-spec.json'),
             dict(operator='expm1_', source=REMOTE + source.relative_to(PROJECT).as_posix(), source_sha256=sha(source),
                  leaf=REMOTE + leaf.relative_to(PROJECT).as_posix(), case=case, reference_kind='original_PyTorch_only'))
    save(BATCH / 'progress.json', dict(status='prepared', accepted=25, total=30,
                                     current_stage='等待最新起点三类型成对采集', started_device_work=False))
    print(json.dumps(dict(status='prepared', accepted=25, batch=BATCH.name)))


if __name__ == '__main__':
    main()
