"""冻结最后两项原合同与已知起点，不改变历史批次。"""
import hashlib
import json
from pathlib import Path
import shutil

PROJECT=Path(__file__).resolve().parents[4]
BATCH='remaining-final-20261031'
OPS=['fix','expm1_']

def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    old=PROJECT/'operators/campaigns/remaining-nine-20261030'
    out=PROJECT/'operators/campaigns'/BATCH
    if out.exists():raise FileExistsError(out)
    plan=json.loads((old/'plan.json').read_text())
    delivery=json.loads((old/'delivery.json').read_text())
    records=[r for r in plan['records'] if r['operator'] in OPS]
    out.mkdir(parents=True)
    for r in records:
        for name in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']:
            src=old/r['catalog_leaf']/name
            assert hashlib.sha256(src.read_bytes()).hexdigest()==r['contract_sha256'][name]
            dst=out/r['catalog_leaf']/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        for rel in [r['seed'],'starting-points/'+r['operator']+'.py']:
            dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/rel,dst)
    bootstrap={op:dict(path='starting-points/'+op+'.py',sha256=hashlib.sha256((out/'starting-points'/ (op+'.py')).read_bytes()).hexdigest()) for op in OPS}
    save(out/'bootstrap.json',bootstrap)
    save(out/'initial-evidence.json',dict(accepted_count=22,total=30,source_delivery=str(old/'delivery.json'),
        source_delivery_sha256=hashlib.sha256((old/'delivery.json').read_bytes()).hexdigest(),
        rows=[r for r in delivery['rows'] if r['operator'] in OPS]))
    save(out/'plan.json',dict(plan,records=records,priority_operators=OPS,verification_operators=OPS,duration_hours=12,
        measurement_policy='新实现一次完整评测及补测；选定源码固定一次确认，低分保留，不反复抽取最好一次'))
    save(out/'targets.json',dict(targets=[dict(order=i+1,operator=op,status='待源码优化') for i,op in enumerate(OPS)]))
    save(out/'catalog/manifest.json',dict(api_version='v6.2',name=BATCH,evaluator='native',layout='per-operator'))
    shutil.copyfile(old/'software-lock.json',out/'software-lock.json')
    save(out/'preparation.json',dict(status='prepared',operators=OPS))
    print(json.dumps(dict(status='prepared',batch=BATCH)))

if __name__=='__main__':main()
