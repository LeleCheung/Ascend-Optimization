"""冻结上一轮25项交付，为剩余五项建立独立原合同批次，不执行设备任务。"""
import hashlib
import json
from pathlib import Path
import shutil

PROJECT=Path(__file__).resolve().parents[4]
PREVIOUS=PROJECT/'operators/campaigns/strict-night-20261102'
BATCH=PROJECT/'operators/campaigns/strict-five-20261103'
REMOTE='/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/'
OPERATORS=['cumulative_trapezoid','uniform','fractional_max_pool3d','expm1_','linalg_inv_ex']
FILES=['definition.json','oracle.py','correctness.jsonl','timing.jsonl']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')


def main():
    if (BATCH/'plan.json').exists():
        value=read(BATCH/'plan.json')
        assert value['starting_delivery_sha256']==sha(PREVIOUS/'delivery.json')
        print('EXISTS_VERIFIED',BATCH);return
    d=read(PREVIOUS/'delivery.json')
    assert d['accepted']==25 and len(d['rows'])==30
    assert read(PREVIOUS/'state.json')['status']=='backend_verified'
    assert read(PREVIOUS/'analysis/final-goal-audit.json')['status']=='verified'
    BATCH.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(PREVIOUS/'delivery.json',BATCH/'starting-delivery.json')
    rows={row['operator']:row for row in d['rows']};records=[]
    for op in OPERATORS:
        old=rows[op];assert not old['accepted'] and not old['cann']
        raw=old['source'];assert raw.startswith(REMOTE)
        source=PROJECT/raw[len(REMOTE):]
        assert sha(source)==old['source_sha256']
        seed=BATCH/'starting-points'/(op+'.py');seed.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,seed)
        origin=PREVIOUS/'original-catalog/ops'/op
        target=BATCH/'catalog/ops'/op
        shutil.copytree(origin,target,ignore=shutil.ignore_patterns('__pycache__'))
        contracts={name:sha(target/name) for name in FILES}
        assert all(contracts[name]==sha(origin/name) for name in FILES)
        records.append(dict(operator=op,seed=seed.relative_to(BATCH).as_posix(),source_sha256=sha(seed),
            original_source=raw,catalog_leaf='catalog/ops/'+op,contract_sha256=contracts,
            reference='original_PyTorch_only',starting_row=old))
    save(BATCH/'catalog/manifest.json',dict(api_version='v6.2',name=BATCH.name,evaluator='native',layout='per-operator'))
    save(BATCH/'plan.json',dict(schema='hgkda-strict-five-v1',server='http://127.0.0.1:19659',physical_card=7,
        timing_scope='walltime',starting_commit='231ba49d69d6df45cfc51e6fd0fb04066b12e58d',
        starting_delivery_sha256=sha(PREVIOUS/'delivery.json'),frozen_accepted=25,target=30,records=records,
        candidate_backend='strict_Triton',reference_policy='合格原生CANN/来源明确Ascend C；保留原PyTorch正确性。未核验原生参考不计达标。',
        measurement_policy='原输入/容差/计时不变；同源码只一次完整评测；未知请求不重发，不修改确定性/范围凑数。',
        passes=[dict(rounds=4,budget_seconds=2400)],profile_budget_seconds=240))
    print(json.dumps(dict(status='prepared',batch=str(BATCH),frozen_accepted=25,operators=OPERATORS)))


if __name__=='__main__':main()
