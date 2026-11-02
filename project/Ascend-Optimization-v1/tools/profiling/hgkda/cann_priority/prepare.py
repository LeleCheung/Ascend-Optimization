"""冻结30项原合同及最近交付，新增CANN对照不覆盖历史。"""
from pathlib import Path
import shutil
from common import BATCH, PROJECT, read, save, sha


def main():
    if (BATCH / 'plan.json').exists():
        raise FileExistsError('批次已经准备，恢复时不能覆盖')
    campaigns = BATCH.parent
    previous = campaigns / 'reference-closure-20261031'
    table = read(previous / '30项验收总表.json')
    plans = [campaigns / n for n in ['reference-closure-20261031', 'threeway-20261030',
                                    'next25-20261029-v2', 'acceptance-three-20261026']]
    indices = [(b, {r['operator']: r for r in read(b / 'plan.json')['records']}) for b in plans]
    records, bootstrap = [], {}
    for row in table['rows']:
        op = row['operator']
        origin, record = next((b, dict(rows[op])) for b, rows in indices if op in rows)
        leaf = origin / record['catalog_leaf']
        if row.get('source'):
            source = Path(row['source'])
            assert sha(source) == row['source_sha256']
        elif op == 'cumulative_trapezoid':
            source = previous / 'proposals/cumulative_trapezoid/hybrid-cann-v3/main.py'
        elif op == 'linalg_inv_ex':
            source = previous / 'proposals/linalg_inv_ex/padded-block-v1/main.py'
        else:
            source = previous / 'starting-points' / (op + '.py')
        # 原始seed留给KG，已交付源码作为本次优化与对照的起点。
        seed = origin / record['seed']
        target = BATCH / 'seeds' / (op + '.py')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(seed.read_bytes())
        frozen = BATCH / 'starting-points' / (op + '.py')
        frozen.parent.mkdir(exist_ok=True)
        frozen.write_bytes(source.read_bytes())
        bootstrap[op] = dict(path=frozen.relative_to(BATCH).as_posix(), sha256=sha(frozen),
                             origin=str(source), previous_regular=row['regular_accepted'])
        hashes = {}
        for name in ['definition.json', 'oracle.py', 'correctness.jsonl', 'timing.jsonl']:
            f = BATCH / 'catalog/ops' / op / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes((leaf / name).read_bytes())
            hashes[name] = sha(f)
        record.update(seed=target.relative_to(BATCH).as_posix(), source_sha256=sha(target),
                      catalog_leaf='catalog/ops/' + op, contract_sha256=hashes)
        records.append(record)
    save(BATCH / 'bootstrap.json', bootstrap)
    save(BATCH / 'plan.json', dict(schema='hgkda-cann-priority-v1', server='http://127.0.0.1:19659',
         physical_card=7, timing_scope='walltime', records=records, duration_hours=16,
         passes=[dict(rounds=3, budget_seconds=2400)], profile_budget_seconds=240,
         measurement_policy='每份源码一次完整KGS及补测；CANN基线、原PyTorch参考、既有源码复测和新优化分列；未知请求不重发'))
    save(BATCH / 'catalog/manifest.json', dict(api_version='v6.2', name=BATCH.name,
         evaluator='native', layout='per-operator'))
    shutil.copyfile(previous / 'software-lock.json', BATCH / 'software-lock.json')
    save(BATCH / 'starting-acceptance.json', dict(previous=str(previous / '30项验收总表.json'),
         sha256=sha(previous / '30项验收总表.json'), regular=24, total=30, table=table))
    print('PREPARED', len(records))


if __name__ == '__main__':
    main()
