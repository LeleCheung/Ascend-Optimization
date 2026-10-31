"""冻结剩余九项的合同和历史合格起点；不覆盖旧批次。"""
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[4]
BATCH = 'remaining-nine-20261030'
OPS = ['negative_', 'expm1_', 'log2_', 'fix', 'round_out',
       'special_round_out', 'hardswish', 'fmod_tensor_', 'linalg_inv_ex']


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def main():
    old = PROJECT/'operators/campaigns/next25-20261029-v2'
    out = PROJECT/'operators/campaigns'/BATCH
    if out.exists():
        raise FileExistsError(out)
    plan = read(old/'plan.json')
    index_path = old/'exports/best-index-final-20261030T004941Z.json'
    entries = {row['operator']: row for row in read(index_path)['operators']}
    inputs = []
    for op in OPS:
        original = next(row for row in plan['records'] if row['operator'] == op)
        seed = old/original['seed']
        assert sha(seed) == original['source_sha256']
        entry = entries[op]
        source = PROJECT/entry['source'] if op != 'linalg_inv_ex' else seed
        assert sha(source) == (entry['source_sha256'] if op != 'linalg_inv_ex' else sha(seed))
        leaf = old/original['catalog_leaf']
        contracts = {name: sha(leaf/name) for name in
                     ['definition.json', 'oracle.py', 'correctness.jsonl', 'timing.jsonl']}
        for name, digest in contracts.items():
            assert not original.get('contract_sha256', {}).get(name) or original['contract_sha256'][name] == digest
        record = dict(original, seed='seeds/'+op+'.py', catalog_leaf='catalog/ops/'+op,
                      contract_sha256=contracts)
        inputs.append((op, seed, source, leaf, record, entry))
    out.mkdir(parents=True)
    bootstrap = {}
    for op, seed, source, leaf, record, entry in inputs:
        target = out/record['seed'];target.parent.mkdir(exist_ok=True)
        target.write_bytes(seed.read_bytes())
        start = out/'starting-points'/(op+'.py');start.parent.mkdir(exist_ok=True)
        start.write_bytes(source.read_bytes())
        for name in record['contract_sha256']:
            target = out/record['catalog_leaf']/name;target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((leaf/name).read_bytes())
        bootstrap[op] = dict(path=start.relative_to(out).as_posix(), sha256=sha(start),
                             origin=source.relative_to(PROJECT).as_posix(), historical=entry)
    save(out/'bootstrap.json', bootstrap)
    save(out/'plan.json', dict(schema_version='hgkda-remaining-v1', server='http://127.0.0.1:19659',
         physical_card=7, timing_scope='walltime', final_repeats=1, duration_hours=18,
         records=[row[4] for row in inputs], passes=[dict(rounds=3, budget_seconds=900)],
         profile_budget_seconds=240, priority_operators=OPS, verification_operators=OPS,
         attribution='Codex诊断与源码改进；需要搜索时CC/DeepSeek生成；固定KGS验收',
         measurement_policy='起点复验；新候选一次完整合同和补测；临界项独立确认，不择最好一次'))
    save(out/'catalog/manifest.json', dict(api_version='v6.2', name=BATCH, evaluator='native', layout='per-operator'))
    save(out/'targets.json', dict(targets=[dict(order=i+1, operator=op, status='待本批复验') for i, op in enumerate(OPS)]))
    shutil.copyfile(old/'software-lock.json', out/'software-lock.json')
    save(out/'preparation.json', dict(status='prepared', operators=OPS, source_index_sha256=sha(index_path)))
    print(json.dumps(dict(status='prepared', batch=BATCH, operators=len(OPS))))


if __name__ == '__main__':
    main()
