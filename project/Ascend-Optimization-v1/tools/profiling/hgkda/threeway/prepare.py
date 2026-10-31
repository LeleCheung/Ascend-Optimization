"""冻结三路对照批次；复用原合同和已交付源码，不覆盖历史实验。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[4]
CURRENT = [
    'selu_', 'rad2deg_', 'uniform', 'value_selecting_reduction_backward',
    'scatter_add', 'upsample_bilinear2d', 'xlogy_scalar_tensor',
    'hardshrink_backward', 'subtract_', 'transform_bias_rescale_qkv',
    'hardswish_out', 'special_xlogy_', 'log10_',
]
PRIORITY = ['log10_', 'cross_out', 'nan_to_num_', 'native_group_norm']


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', default='threeway-20261030')
    a = p.parse_args()
    if not a.batch.replace('-', '').isalnum():
        raise ValueError('非法批次名')
    campaigns = PROJECT/'operators/campaigns'
    out = campaigns/a.batch
    if out.exists():
        raise FileExistsError(out)
    old = campaigns/'next25-20261029-v2'
    index = read(old/'exports/best-index-final-20261030T004941Z.json')
    entries = {r['operator']: r for r in index['operators']}
    sources = {}
    for op in set(CURRENT+PRIORITY)-{'scatter_add', 'upsample_bilinear2d', 'value_selecting_reduction_backward'}:
        entry = entries[op]
        source = PROJECT/entry['source']
        assert sha(source) == entry['source_sha256']
        sources[op] = (old, source, entry)
    for op, folder in [
        ('value_selecting_reduction_backward', 'night-20261021-v3-value-stride-repair'),
        ('scatter_add', 'remaining-five-20261023-v2'),
        ('upsample_bilinear2d', 'two-closure-20261024'),
    ]:
        previous = campaigns/folder
        source = previous/'deliverables/reviewed'/op/'main.py'
        if op == 'value_selecting_reduction_backward':
            source = previous/'seeds'/ (op+'.py')
        if not source.is_file():
            raise FileNotFoundError(source)
        sources[op] = (previous, source, dict(source=source.relative_to(PROJECT).as_posix(), source_sha256=sha(source)))
    # 全部输入核验完成后才创建目录，失败不留下半份待运行合同。
    plans = {str(previous): read(previous/'plan.json') for previous, _, _ in sources.values()}
    records = []
    for op in list(dict.fromkeys(PRIORITY+CURRENT)):
        previous, source, entry = sources[op]
        original = next(r for r in plans[str(previous)]['records'] if r['operator'] == op)
        leaf = previous/original['catalog_leaf']
        frozen = dict(original, seed='seeds/'+op+'.py', catalog_leaf='catalog/ops/'+op)
        for key in ['upstream_source']:
            if key in frozen:
                frozen[key] = frozen[key].replace('\\', '/')
        # seed始终代表原版/历史起点原件；bootstrap单独登记最新有效源码。
        baseline = previous/original['seed']
        assert sha(baseline) == original['source_sha256']
        files = {}
        for name in ['definition.json', 'oracle.py', 'correctness.jsonl', 'timing.jsonl']:
            digest = sha(leaf/name)
            if original.get('contract_sha256', {}).get(name):
                assert digest == original['contract_sha256'][name]
            files[name] = digest
        frozen['contract_sha256'] = files
        records.append((op, previous, source, entry, frozen))
    out.mkdir(parents=True)
    bootstrap = {}
    for op, previous, source, entry, record in records:
        for name in record['contract_sha256']:
            dest = out/record['catalog_leaf']/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((previous/next(r for r in plans[str(previous)]['records'] if r['operator']==op)['catalog_leaf']/name).read_bytes())
        dest = out/record['seed']
        dest.parent.mkdir(parents=True, exist_ok=True)
        original = next(r for r in plans[str(previous)]['records'] if r['operator']==op)
        dest.write_bytes((previous/original['seed']).read_bytes())
        target = out/'starting-points'/(op+'.py')
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(source.read_bytes())
        bootstrap[op] = dict(path=target.relative_to(out).as_posix(), sha256=sha(target),
                             origin=source.relative_to(PROJECT).as_posix(), historical=entry)
    plan = dict(schema_version='hgkda-threeway-v1', server='http://127.0.0.1:19659',
                physical_card=7, timing_scope='walltime', final_repeats=1, duration_hours=18,
                records=[row[-1] for row in records], passes=[dict(rounds=5, budget_seconds=2700)],
                profile_budget_seconds=240, verification_operators=CURRENT, priority_operators=PRIORITY,
                measurement_policy='默认一次完整KGS；临界项同窗口预先固定轮次聚合，不择最好一次',
                attribution='Codex编排与机制诊断；CC/DeepSeek生成；固定KGS评测')
    save(out/'plan.json', plan)
    save(out/'bootstrap.json', bootstrap)
    save(out/'catalog/manifest.json',dict(api_version='v6.2',name=a.batch,evaluator='native',layout='per-operator'))
    save(out/'targets.json', dict(targets=[dict(order=i+1, operator=row[0], status='待本批复验') for i, row in enumerate(records)]))
    shutil.copyfile(old/'software-lock.json', out/'software-lock.json')
    save(out/'preparation.json', dict(status='prepared', operators=len(records),
         source_index_sha256=sha(old/'exports/best-index-final-20261030T004941Z.json'),
         plan_sha256=sha(out/'plan.json')))
    print(json.dumps(dict(status='prepared', path=str(out), operators=len(records)), ensure_ascii=False))


if __name__ == '__main__':
    main()
