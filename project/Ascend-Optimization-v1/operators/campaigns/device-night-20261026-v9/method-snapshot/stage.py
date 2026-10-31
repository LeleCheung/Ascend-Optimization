"""开发阶段：持卡锁取得父版KGS与设备机制干预证据；独立集暂不运行。"""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import shlex
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('device_stage_queue',HERE.parent/'run-nightly.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


def stage(batch,root):
    q=queue.Queue(batch,root,'tle_yy');state=dict(status='running',started_at=queue.now(),current=None,records=[])
    out=q.batch;queue.save(out/'stage-state.json',state)
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for rec in q.plan['records']:
                op=rec['operator'];state['current']=op;queue.save(out/'stage-state.json',state)
                directory=out/'results'/op/'development';source=out/rec['seed']
                q.idle();q.run('device-parent-semantics-'+op,shlex.join([q.py,str(HERE/'check.py'),'--operator',op,
                    '--source',str(source),'--seed',str(source),'--output',str(directory/'parent-semantics'),'--semantic-only']),900)
                baseline=q.evaluate(op,source,directory/'parent','parent',1)
                queue.save(directory/'baseline-summary.json',baseline)
                case_dir=out/'probe-inputs'/op;case_dir.mkdir(parents=True)
                oracle=(out/rec['catalog_leaf']/'oracle.py').read_bytes();(case_dir/'oracle.py').write_bytes(oracle)
                cases=([dict(shape=[1024,1024,1024],dtype=d) for d in ['float16','float32']] if op=='matmul_bias_activation' else
                       [dict(shape=s,dtype=d,dim=1,keepdim=False) for s in [[4096,4096],[64,512,512]] for d in ['float16','float32']])
                queue.save(case_dir/'cases.json',cases);queue.save(case_dir/'configs.json',rec['probes'])
                q.idle();start=time.monotonic()
                q.run('device-probe-'+op,shlex.join([q.py,str(HERE/'probe.py'),'--operator',op,'--source',str(source),
                    '--configs',str(case_dir/'configs.json'),'--cases',str(case_dir/'cases.json'),'--output',str(directory/'interventions')]),7200)
                state['records'].append(dict(operator=op,status='completed',probe_seconds=time.monotonic()-start))
                queue.save(out/'stage-state.json',state)
            state.update(status='completed',current=None,finished_at=queue.now())
        except BaseException as error:state.update(status='needs_attention',error=str(error));raise
        finally:queue.save(out/'stage-state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True)
    p.add_argument('--root',type=Path,default=Path('/data/hanle/ascend-optimization'));a=p.parse_args();stage(a.batch,a.root)
