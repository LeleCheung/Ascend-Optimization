"""在同一项目锁内补齐最终混合候选的末轴/非末轴实际设备路径与IR。"""
import fcntl
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import run

def main():
    with (run.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);r=run.Runner(run.BATCH.name);op='cumulative_trapezoid'
        source=run.BATCH/'proposals'/op/'hybrid-cann-v3/main.py';leaf=run.BATCH/r.q.records[op]['catalog_leaf']
        cases=[json.loads(t) for t in (leaf/'timing.jsonl').read_text().splitlines() if t.strip()]
        selected=[next(c for c in cases if c['inputs']['spec']['dtype']=='float16' and c['inputs']['spec']['dim']==0),
                  max((c for c in cases if c['inputs']['spec']['dtype']=='float16' and c['inputs']['spec']['dim']==-1),key=lambda c:__import__('math').prod(c['inputs']['spec']['shape']))]
        for case in selected:
            out=run.BATCH/'paths'/op/('hybrid-v3-'+case['name']);out.mkdir(parents=True,exist_ok=True)
            spec=out/'spec.json';value=dict(operator=op,leaf=str(leaf),source=str(source),source_sha256=run.sha(source),case=case,deterministic=True)
            if spec.exists():assert run.read(spec)==value
            else:run.base.save_json(spec,value)
            for side in ['candidate','pytorch']:
                for kind in (['trace','ir'] if side=='candidate' else ['trace']):
                    target=out/(side+'-'+kind+('.json' if kind=='trace' else ''))
                    wrapper=out/(side+'-'+kind+'-driver.py')
                    args=[str(run.NEXT/'driver.py'),'--spec',str(spec),'--side',side,'--'+kind,str(target)]
                    wrapper.write_text('import torch,torch_npu,runpy,sys'+chr(10)+'torch.use_deterministic_algorithms(True)'+chr(10)+'sys.argv='+repr(args)+chr(10)+'runpy.run_path('+repr(str(run.NEXT/'driver.py'))+',run_name='+repr('__main__')+')'+chr(10))
                    def capture(target=target,wrapper=wrapper,side=side,kind=kind,out=out):
                        if not target.exists():r._cmd('scan-final-'+case['name']+'-'+side+'-'+kind,[run.base.PY,wrapper],600)
                        if kind=='trace':
                            parser=run.load('scan_final_trace',run.THREE/'trace_summary.py');summary=parser.summarize(target);run.base.save_json(out/(side+'-summary.json'),summary)
                            return {k:v for k,v in summary.items() if k!='device_tasks'}
                        manifest=run.read(target/'manifest.json');assert manifest['entries'],'未捕获实际Triton编译产物'
                        return dict(status='captured',kernels=[e['symbol'] for e in manifest['entries']],manifest=str(target/'manifest.json'),sha256=run.sha(target/'manifest.json'))
                    r.job('scan-final-evidence:'+case['name']+':'+side+':'+kind,capture)
        r.state.update(status='completed_with_reference_status',current_job=None,finished_at=run.now());r.save()
        r._cmd('scan-final-report',[run.base.PY,HERE/'report_v2.py'],180)

if __name__=='__main__':main()
