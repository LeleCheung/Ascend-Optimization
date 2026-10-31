"""在新目录复现机制变换交付：不调用模型、不改旧证据，一次补充检查和完整KGS。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mechanism_reproduce_queue',HERE/'run-nightly.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',required=True);p.add_argument('--operator',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/data/hanle/ascend-optimization'))
    a=p.parse_args()
    output=a.output.resolve();output.relative_to((a.root/'runs').resolve())
    if output.exists():raise ValueError('复现输出已存在')
    q=m.Queue(a.batch,a.root,'tle_yy')
    record=q.records[a.operator];source=q.batch/'results'/a.operator/'mechanism_rewrite/candidate.py'
    state=json.loads((q.batch/'state.json').read_text())
    entry=next(r for r in state['records'] if r['operator']==a.operator and r['arm']=='mechanism_rewrite')
    if entry['status']!='passed' or hashlib.sha256(source.read_bytes()).hexdigest()!=entry['source_sha256']:
        raise ValueError('没有已完整通过且身份正确的交付')
    snapshot=q.batch/'method-snapshot'
    if hashlib.sha256((snapshot/'check.py').read_bytes()).hexdigest()!=q.plan['method_sha256']['check.py']:
        raise ValueError('冻结检查器身份不同')
    # 复现日志进入新目录，不追加旧批次证据。
    q.runtime=output/'host-logs';q.runtime.mkdir(parents=True)
    status=dict(status='queued',operator=a.operator,source_sha256=entry['source_sha256'],batch=a.batch)
    m.save(output/'state.json',status)
    with (a.root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            q.idle();status['status']='running';m.save(output/'state.json',status)
            q.run('check-'+a.operator,shlex.join([q.py,str(snapshot/'check.py'),'--operator',a.operator,
                '--source',str(source),'--seed',str(q.batch/record['seed']),'--output',str(output/'semantics.json')]),1800)
            status['formal']=q.evaluate(a.operator,source,output/'formal','independent-reproduction',1)
            status['status']='completed'
        except BaseException as e:status.update(status='needs_attention',error=str(e));raise
        finally:m.save(output/'state.json',status)


if __name__=='__main__':main()
