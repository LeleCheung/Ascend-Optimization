"""冻结六项未完成对象及两个新达标对象，原合同逐字节保留。"""
import hashlib
import json
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
BATCH=PROJECT/'operators/campaigns/reference-closure-20261031'
NAMES=['linalg_inv_ex','fake_quantize_per_tensor_affine_cachemask_out','batch_norm_with_update_functional','cumulative_trapezoid','fractional_max_pool3d','_conj_physical','fix','expm1_']
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    if (BATCH/'plan.json').exists():raise FileExistsError(BATCH/'plan.json')
    c=PROJECT/'operators/campaigns';new=c/'next25-20261029-v2';old=c/'acceptance-three-20261026';last=c/'remaining-final-20261031'
    records=[];boot={}
    newest=read(last/'delivery.json');assert newest['targets']['regular_accepted']==24
    indices={r['operator']:r for r in read(new/'plan.json')['records']}
    exports={r['operator']:r for r in read(new/'exports/best-index-final-20261030T004941Z.json')['operators']}
    oldplan={r['operator']:r for r in read(old/'plan.json')['records']}
    selected={r['operator']:r for r in newest['rows']}
    for op in NAMES:
        if op in oldplan:
            record=dict(oldplan[op]);leaf=old/record['catalog_leaf'];seed=old/'deliverables'/op/'main.py';source=seed
        else:
            record=dict(indices[op]);leaf=new/record['catalog_leaf'];seed=new/record['seed']
            source=seed if op=='linalg_inv_ex' else Path(selected[op]['source']) if op in selected else PROJECT/exports[op]['source']
        record.update(seed='seeds/'+op+'.py',source_sha256=sha(seed),catalog_leaf='catalog/ops/'+op,
            contract_sha256={n:sha(leaf/n) for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']})
        for n,d in record['contract_sha256'].items():
            f=BATCH/record['catalog_leaf']/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((leaf/n).read_bytes());assert sha(f)==d
        f=BATCH/record['seed'];f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(seed.read_bytes())
        f=BATCH/'starting-points'/(op+'.py');f.parent.mkdir(exist_ok=True);f.write_bytes(source.read_bytes())
        boot[op]=dict(path=f.relative_to(BATCH).as_posix(),sha256=sha(f),origin=str(source))
        records.append(record)
    save(BATCH/'bootstrap.json',boot)
    save(BATCH/'plan.json',dict(schema='hgkda-reference-closure-v1',server='http://127.0.0.1:19659',physical_card=7,timing_scope='walltime',
        records=records,duration_hours=16,passes=[dict(rounds=3,budget_seconds=2400)],profile_budget_seconds=240,
        measurement_policy='每个新源码一次完整KGS；既有24项不重搜；原参考与新参考分列；未知请求不重发'))
    save(BATCH/'targets.json',dict(targets=[dict(order=i+1,operator=op,status='待资格核验') for i,op in enumerate(NAMES)]))
    save(BATCH/'catalog/manifest.json',dict(api_version='v6.2',name=BATCH.name,evaluator='native',layout='per-operator'))
    shutil.copyfile(last/'software-lock.json',BATCH/'software-lock.json')
    save(BATCH/'starting-acceptance.json',dict(previous_delivery=str(last/'delivery.json'),sha256=sha(last/'delivery.json'),accepted=newest['accepted'],regular_accepted=24,total=30))
    print(json.dumps(dict(status='prepared',operators=NAMES)))
if __name__=='__main__':main()
