#!/usr/bin/env python3
"""等待前序终态后，补快路径门禁、导出兼容采集的源码并采集最终成对证据。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import time


def quote(args):
    return ' '.join(shlex.quote(str(x)) for x in args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suffix', default='profilemeta-v3')
    p.add_argument('--wait-suffixes', nargs='+')
    p.add_argument('--arms', nargs='+')
    p.add_argument('--batch', default='paired-pointwise-20261022')
    p.add_argument('--wait-kg', action='store_true', help='等待本批KG终态，再检查候选队列')
    p.add_argument('--skip-backfill', action='store_true', help='本批各来源已完成快路径门禁，不补旧批次来源')
    a = p.parse_args()
    if not a.suffix.replace('-', '').isalnum():
        raise ValueError('非法候选版本')
    if not a.batch.replace('-', '').replace('_', '').isalnum():
        raise ValueError('非法批次名')
    tools = Path(__file__).resolve().parent
    root = Path('/data/hanle/ascend-optimization')
    s = importlib.util.spec_from_file_location('paired_finish_queue', tools/'run-nightly.py')
    m = importlib.util.module_from_spec(s);s.loader.exec_module(m)
    q = m.Queue(a.batch,root,'tle_yy')
    additional = q.batch/'additional-candidate-queues.json'
    if additional.exists():
        extra=json.loads(additional.read_text())
        suffixes=extra['suffixes']
        if not all(s.replace('-', '').isalnum() for s in suffixes):
            raise ValueError('非法追加候选版本')
        a.wait_suffixes=list(dict.fromkeys((a.wait_suffixes or [a.suffix])+suffixes))
        a.arms=list(dict.fromkeys((a.arms or ['codex-'+a.suffix])+['codex-'+s for s in suffixes]))
    path = q.batch/('completion-state-'+a.suffix+'.json')
    if path.exists():
        raise ValueError('终验已有状态，不重复提交设备请求')
    state = dict(status='waiting', results={}, candidate_suffix=a.suffix)
    m.save(path,state)
    # 等待不占卡锁，避免抢在既有排队任务之前导致互相阻塞。
    deadline = time.monotonic()+14400
    try:
        while True:
            values=[]
            if a.wait_kg:
                kg_path = q.batch/'kg-paired-state.json'
                kg = json.loads(kg_path.read_text()) if kg_path.exists() else {}
                if kg.get('status') == 'needs_attention':
                    raise RuntimeError('前序KG控制器异常，先检查原始日志')
                values.append(kg.get('status') == 'completed')
            for suffix in (a.wait_suffixes or [a.suffix]):
                candidate = q.batch/('candidate-state-'+suffix+'.json')
                if candidate.exists():
                    value = json.loads(candidate.read_text())
                    if value['status'] == 'needs_attention':
                        raise RuntimeError('前序候选失败，先检查原始日志：'+suffix)
                    values.append(value['status']=='completed')
                else:values.append(False)
            if all(values):break
            if time.monotonic()>deadline:
                raise RuntimeError('前序等待超过预算；不重发设备请求')
            time.sleep(10)
        import fcntl
        with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            q.idle()
            state['status']='running';m.save(path,state)
            # 为原成绩补独立快路径门禁，不重新跑其正式性能，保留一次完整评测口径。
            for op in ['log10_','selu_']:
                for arm in ([] if a.skip_backfill else ['codex-rawptr-v1','paired-HG-KDA']):
                    directory=q.batch/'results'/op/arm
                    semantic=directory/'semantics.json'
                    if not semantic.exists() or json.loads(semantic.read_text())['status']!='passed':
                        continue
                    source=(q.batch/'candidates'/(op+'-rawptr-v1.py') if arm.startswith('codex-')
                            else directory/'export/main.py')
                    output=directory/'dense-semantics.json'
                    if not output.exists():
                        q.run('dense-'+op+'-'+arm,quote([q.py,tools/'check-dense-pointwise.py',
                            '--operator',op,'--source',source,'--output',output]),1800)
            delivery=q.batch/('deliverables-'+a.suffix)
            range_policy = q.batch/'require-range-gate.json'
            if range_policy.exists():
                policy = json.loads(range_policy.read_text())
                checker = tools/'check-pointwise-ranges.py'
                if policy['checker_sha256'] != hashlib.sha256(checker.read_bytes()).hexdigest():
                    raise ValueError('冻结数值范围门禁与检查器身份不同')
                for op in ['log10_', 'selu_']:
                    for arm in (a.arms or ['codex-'+a.suffix]):
                        directory = q.batch/'results'/op/arm
                        dense = directory/'dense-semantics.json'
                        if not dense.exists() or json.loads(dense.read_text())['status'] != 'passed':
                            continue
                        source = (q.batch/'candidates'/(op+'-'+arm[len('codex-'):]+'.py')
                                  if arm.startswith('codex-') else directory/'export/main.py')
                        ranges = directory/'range-semantics.json'
                        if not ranges.exists():
                            try:
                                q.run('ranges-'+op+'-'+arm, quote([q.py,checker,'--operator',op,
                                    '--source',source,'--output',ranges]),1800)
                            except RuntimeError:
                                if not ranges.exists():raise
            # 本次正式交付要求元数据适配版本；旧的裸地址版本保留为历史性能证据。
            q.run('export-final-'+a.suffix,quote([q.py,tools/'export-paired-deliverables.py',
                '--batch-dir',q.batch,'--output',delivery,'--require-dense','--arms']
                +(a.arms or ['codex-'+a.suffix])),600)
            records=json.loads((delivery/'manifest.json').read_text())['records']
            research=q.batch/('research/final-'+a.suffix)
            research.mkdir(parents=True,exist_ok=False)
            for record in records:
                op=record['operator'];source=Path(record['delivered'])
                state['results'][op]=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),pairs=[])
                selected=record['selected'];formal=Path(selected['evidence']['result']).parent
                # msprof op与KGS的msprof基础采集是不同入口，两者都验证。
                q.run('final-kgs-profile-'+op+'-'+a.suffix,quote([q.py,
                    q.project/'tools/evaluation/profile-operator.py',
                    '--evaluation',selected['evidence']['result'],'--request',selected['evidence']['request'],
                    '--inspect',formal/'inspect.json','--case-id',op+'-timing-3',
                    '--output',research/(op+'-kgs-metrics'),'--server',q.server,'--level','metrics',
                    '--aic-metrics','PipeUtilization','--timeout','600']),900)
                state['results'][op]['kgs_profile']=str(research/(op+'-kgs-metrics/response.json'))
                m.save(path,state)
                for index in [0,3]:
                    original=q.batch/'paired-specs'/('%s-%d.json'%(op,index))
                    spec=json.loads(original.read_text())
                    spec.update(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                    spec_path=research/('%s-%d-spec.json'%(op,index));m.save(spec_path,spec)
                    for label,active_spec in ([('seed',original),('final',spec_path)] if index==3 else []):
                        q.run('final-ir-'+op+'-'+label+'-'+a.suffix,quote([q.py,tools/'paired-driver.py',
                            '--spec',active_spec,'--side','candidate','--dump-ir',research/(op+'-'+label+'-ir')]),900)
                    output=research/('%s-paired-%d'%(op,index))
                    state['current_job']=output.name;m.save(path,state)
                    q.run('final-paired-'+op+'-'+str(index)+'-'+a.suffix,quote([q.py,tools/'collect-paired.py',
                        '--spec',spec_path,'--output',output]),2400)
                    q.run('final-compare-'+op+'-'+str(index)+'-'+a.suffix,quote([q.py,
                        tools/'skills/hgkda/scripts/compare_opprof.py','--directory',output,
                        '--output',output/'comparison.json']),600)
                    state['results'][op]['pairs'].append(str(output/'comparison.json'));m.save(path,state)
            state.update(status='completed',current_job=None)
    except Exception as error:
        state.update(status='needs_attention',error=str(error))
        raise
    finally:
        m.save(path,state)


if __name__=='__main__':
    main()
