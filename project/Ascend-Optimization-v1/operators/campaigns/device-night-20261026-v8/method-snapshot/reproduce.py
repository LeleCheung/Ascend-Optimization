"""在910B新目录复现冻结的设备机制交付，不重新调用模型或调参。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('device_reproduce_queue',HERE.parent/'run-nightly.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True)
    p.add_argument('--operator',choices=['amin','matmul_bias_activation'],required=True)
    p.add_argument('--arm',choices=['mechanism','raw_evidence'],default='mechanism')
    p.add_argument('--split',choices=['heldout','development'],default='heldout')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/data/hanle/ascend-optimization'));a=p.parse_args()
    output=a.output.resolve();output.relative_to((a.root/'runs').resolve())
    if output.exists():raise ValueError('请使用新目录')
    q=queue.Queue(a.batch,a.root,'tle_yy');frozen=json.loads((q.batch/'selection-frozen.json').read_text())
    selected=frozen['selected'][a.operator+'/'+a.arm];source=q.batch/selected['source']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=selected['source_sha256']:raise ValueError('交付源码身份不同')
    snapshot=q.batch/'system-snapshot';method=json.loads((q.batch/'frozen-method.json').read_text())
    for name,digest in method['files'].items():
        if hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=digest:raise ValueError('冻结方法身份不同：'+name)
    contract=q.records[a.operator]['splits'][a.split]
    for name,digest in contract['sha256'].items():
        if hashlib.sha256((q.batch/contract['path']/name).read_bytes()).hexdigest()!=digest:raise ValueError('合同身份不同')
    q.records[a.operator]['catalog_leaf']=contract['path']
    q.runtime=output/'host-logs';q.runtime.mkdir(parents=True)
    state=dict(status='running',batch=a.batch,operator=a.operator,arm=a.arm,split=a.split,source_sha256=selected['source_sha256'])
    queue.save(output/'state.json',state)
    with (a.root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            q.idle();q.run('supplementary',shlex.join([q.py,str(snapshot/'check.py'),'--operator',a.operator,
                '--source',str(source),'--seed',str(q.batch/q.records[a.operator]['seed']),'--output',str(output/'supplementary')]),3600)
            state['formal']=q.evaluate(a.operator,source,output/'formal','reproduction',1)
            state['status']='completed'
        except BaseException as error:state.update(status='needs_attention',error=str(error));raise
        finally:queue.save(output/'state.json',state)


if __name__=='__main__':main()
