#!/usr/bin/env python3
"""在相同910B环境独立复现已交付两项源码；新目录，42＋12项及一次正式KGS。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',default='paired-pointwise-20261022')
    p.add_argument('--tag',default='final-v6')
    a=p.parse_args()
    if (not a.batch.replace('-', '').replace('_', '').isalnum()
            or not a.tag.replace('-', '').isalnum()):raise ValueError('非法批次或最终版本名')
    root=Path('/data/hanle/ascend-optimization')
    output=a.output.resolve()
    output.relative_to(root/'runs')
    if output.exists():raise ValueError('复现目录已存在')
    tools=Path(__file__).resolve().parent
    s=importlib.util.spec_from_file_location('paired_reproduce',tools/'run-nightly.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    q=m.Queue(a.batch,root,'tle_yy')
    manifest=json.loads((q.batch/('deliverables-'+a.tag)/'manifest.json').read_text())
    output.mkdir(parents=True)
    state=dict(status='queued',results={});m.save(output/'state.json',state)
    import fcntl
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            for record in manifest['records']:
                op=record['operator']
                relative=Path(record['delivered']).relative_to(q.batch)
                source=q.batch/relative
                if hashlib.sha256(source.read_bytes()).hexdigest()!=record['selected']['source_sha256']:
                    raise ValueError('交付源码身份不符')
                q.idle();state.update(status='running',current_operator=op);m.save(output/'state.json',state)
                checkers=[('check-pointwise-semantics.py','semantics.json'),('check-dense-pointwise.py','dense-semantics.json')]
                if (q.batch/'require-range-gate.json').exists():
                    policy=json.loads((q.batch/'require-range-gate.json').read_text())
                    checker=tools/'check-pointwise-ranges.py'
                    if hashlib.sha256(checker.read_bytes()).hexdigest()!=policy['checker_sha256']:
                        raise ValueError('数值范围检查器与冻结门禁不同')
                    checkers.append(('check-pointwise-ranges.py','range-semantics.json'))
                for script,name in checkers:
                    command=[q.py,tools/script,'--operator',op,'--source',source,'--output',output/op/name]
                    q.run('reproduce-'+output.name+'-'+op+'-'+script,
                          ' '.join(shlex.quote(str(x)) for x in command),1800)
                state['results'][op]=q.evaluate(op,source,output/op/'formal','independent-reproduction',1)
                m.save(output/'state.json',state)
            state.update(status='completed',current_operator=None)
        except Exception as error:
            state.update(status='needs_attention',error=str(error));raise
        finally:m.save(output/'state.json',state)


if __name__=='__main__':main()
