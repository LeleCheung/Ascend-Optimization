"""对最终最佳源码做一次新的完整评测与补测；使用独立run-name，不改历史结果。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator',required=True)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--batch',default='remaining-nine-20261030')
    a=parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',a.run_name):raise ValueError('非法run-name')
    here=Path(__file__).resolve().parent
    import sys
    sys.path.insert(0,str(here))
    spec=importlib.util.spec_from_file_location('remaining_reproduction_controller',here/'run.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    with (controller.base.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        r=controller.Runner(a.batch)
        if a.operator not in r.q.records:raise ValueError('本批未登记的算子')
        best=r.state['operators'].get(a.operator,{}).get('best') or {}
        if controller.base.extra_qualification(best)!='passed':raise ValueError('尚无合格源码可复现')
        source=Path(best['source'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=best['source_sha256']:raise ValueError('源码身份不同')
        out=r.batch/'reproductions'/a.run_name/a.operator
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True)
        result=r.evaluate(a.operator,source,out/'formal','HG-KDA-reproduction')
        if result.get('status')=='PASSED':
            gates=out/'extra-checks.json'
            result.update(extra_checks=str(gates),supplementary_status=r.supplementary(a.operator,source,gates,'reproduction-'+a.operator+'-'+a.run_name))
            if best.get('native_manifest'):
                safety=out/'native-safety.json'
                try:
                    r.q.run('reproduction-native-'+a.operator+'-'+a.run_name,controller.base.quote([
                        controller.base.PY,here/'check_native.py','--operator',a.operator,'--source',source,
                        '--leaf',r.batch/r.q.records[a.operator]['catalog_leaf'],'--output',safety]),600)
                except RuntimeError:
                    if not safety.exists():raise
                checked=json.loads(safety.read_text(encoding='utf8'));result['native_safety']=str(safety)
                if checked['status']!='passed' or checked['source_sha256']!=best['source_sha256']:
                    result['supplementary_status']='native_safety_failed'
        result.update(operator=a.operator,scope='一次新完整KGS及独立语义补测；不改历史最佳')
        controller.base.save_json(out/'reproduction.json',result)
        r.wait_idle()
        print(json.dumps({k:result.get(k) for k in ['operator','status','geo_mean','num_passed','num_workloads','supplementary_status','result']},ensure_ascii=False))
        if result.get('status')!='PASSED' or result.get('supplementary_status')!='passed':raise SystemExit(1)


if __name__=='__main__':main()
