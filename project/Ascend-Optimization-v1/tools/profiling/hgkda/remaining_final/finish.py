"""不再重测：读取固定确认，做冻结新形状、真实路径与诊断证据收尾。"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'remaining'))
s=importlib.util.spec_from_file_location('remaining_final_previous',HERE.parent/'remaining/run.py')
previous=importlib.util.module_from_spec(s);s.loader.exec_module(previous)

def main():
    with (previous.base.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        r=previous.Runner('remaining-final-20261031');r.state['active_started_epoch']=time.time()
        try:
            for op in ['fix','expm1_']:
                best=r.state['operators'][op]['best'];source=Path(best['source']);digest=hashlib.sha256(source.read_bytes()).hexdigest()
                assert digest==best['source_sha256']
                confirmation=r.batch/'reproductions/final-confirm-01'/op/'reproduction.json'
                confirmed=json.loads(confirmation.read_text())
                assert confirmed['source_sha256']==digest and confirmed['supplementary_status']=='passed'
                out=r.batch/'frozen-shapes'/op/'checks.json'
                def checks(op=op,source=source,out=out):
                    if not out.exists():
                        try:r.q.run('frozen-shapes-'+op,previous.base.quote([previous.base.PY,HERE/'check_frozen_shapes.py','--operator',op,'--source',source,'--output',out]),900)
                        except RuntimeError:
                            if not out.exists():raise
                    value=json.loads(out.read_text());assert value['source_sha256']==digest
                    return dict(path=str(out),status=value['status'],passed=value['passed'],total=value['total'])
                r.job('frozen-shapes:'+op+':'+digest[:12],checks)
                row=r.state['operators'][op];row['final_confirmation']=confirmed;row['frozen_shapes']=str(out);r.save()
            # 原收集器固定小/大FP32输入，同输入采集起点、PyTorch、当前最佳；任务持同一卡锁串行。
            evidence=previous.base.module('remaining_final_evidence',HERE.parent/'remaining/evidence.py')
            evidence.collect(r)
            for op in ['fix','expm1_']:
                best=r.state['operators'][op]['best'];source=Path(best['source'])
                out=r.batch/'diagnosis'/op/'final.json'
                if not out.exists():
                    import subprocess
                    cmd=[sys.executable,str(HERE.parent/'skills/hgkda/scripts/analyze_execution_paths.py'),
                        '--result',best['result'],'--source',str(source),'--catalog',str(r.batch/r.q.records[op]['catalog_leaf']),
                        '--parent-result',str(r.batch/'proposals'/op/'event-large-v6/assessment/formal/HG-KDA-event-large-v6-1.result.json'),
                        '--paths',str(r.batch/'paths'/op),'--output',str(out)]
                    subprocess.run(cmd,check=True)
            r.wait_idle();r.state.update(status='evidence_completed',current_job=None,finished_at=r.module.now());r.save()
        except Exception as e:
            r.state.update(status='needs_attention',error=type(e).__name__+': '+str(e));r.save();raise

if __name__=='__main__':main()
