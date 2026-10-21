#!/usr/bin/env python3
"""等待本项目卡锁，独立验证已登记的语义修复；不替换原始KG轮次。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

spec=importlib.util.spec_from_file_location('nightly',Path(__file__).with_name('run-nightly.py'))
nightly=importlib.util.module_from_spec(spec);spec.loader.exec_module(nightly)


def valid_dense_semantics(data,source_sha,checker_sha):
    rows=data.get('cases',[])
    expected={(dtype,tuple(sizes),keep) for dtype in ['torch.float16','torch.bfloat16','torch.float32']
              for sizes in [[1,8209],[3,16387],[2,7,8209]] for keep in [False,True]}
    actual={(r.get('dtype'),tuple(r.get('sizes',[])),r.get('keepdim')) for r in rows}
    return (data.get('status')=='passed' and data.get('source_sha256')==source_sha and
            data.get('checker_sha256')==checker_sha and data.get('physical_device')==7 and
            data.get('logical_device')=='npu:0' and data.get('passed')==data.get('total')==18 and
            len(rows)==18 and actual==expected and
            all(r.get('status')=='passed' and r.get('repeats')==5 for r in rows))


def execute(queue):
    mode=queue.plan.get('execution_mode')
    if mode not in ['semantic_repair','moe_semantic_repair']:
        raise ValueError('只接受单独登记的语义修复批次')
    op='moe_sum_reduce' if mode=='moe_semantic_repair' else 'value_selecting_reduction_backward'
    if set(queue.records)!={op}:raise ValueError('修复批次必须只含登记的算子')
    key='semantic-repair:'+op
    def verify():
        queue.idle()
        source=queue.batch/queue.records[op]['seed']
        output=queue.batch/'results'/op/'codex-stride-repair'
        output.mkdir(parents=True,exist_ok=True)
        semantics=output/'supplementary-semantics.json'
        checker=queue.project/'tools/profiling/hgkda/check-nightly-semantics.py'
        if mode=='moe_semantic_repair' and hashlib.sha256(checker.read_bytes()).hexdigest()!=queue.plan['checker_sha256']:
            raise RuntimeError('MoE检查器与冻结plan不同')
        if semantics.exists():
            value=json.loads(semantics.read_text())
            if value.get('status')!='passed':
                raise RuntimeError('已有失败补测，保留证据并使用新尝试目录')
        else:
            args=[queue.py,str(checker),
                  '--operator',op,'--source',str(source),'--output',str(semantics)]
            queue.run(key+':semantics',' '.join(shlex.quote(x) for x in args),budget=3600)
        value=json.loads(semantics.read_text())
        if (value.get('source_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest() or
            value.get('checker_sha256')!=hashlib.sha256(checker.read_bytes()).hexdigest()):
            raise RuntimeError('补测与源码或检查器身份不同')
        expected=36 if mode=='moe_semantic_repair' else 48
        if value.get('status')!='passed' or value.get('passed')!=expected or value.get('total')!=expected:
            raise RuntimeError('%d项补充语义尚未全部通过'%expected)
        if mode=='moe_semantic_repair':
            spec=importlib.util.spec_from_file_location('moe_semantic_report',Path(__file__).with_name('report-nightly.py'))
            report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
            if not report.valid_semantics(value,op,hashlib.sha256(source.read_bytes()).hexdigest(),queue.plan['checker_sha256']):
                raise RuntimeError('MoE语义case集合或设备身份不同')
        extra=queue.plan.get('additional_semantics')
        if extra:
            script=queue.project/extra['script'];script.resolve().relative_to(queue.project.resolve())
            if script.name!='check-value-dense-tiles.py':raise ValueError('未知的额外检查器')
            if hashlib.sha256(script.read_bytes()).hexdigest()!=extra['script_sha256']:
                raise RuntimeError('额外检查器与冻结plan不同')
            target=output/'dense-tile-semantics.json'
            if not target.exists():
                args=[queue.py,str(script),'--source',str(source),'--output',str(target)]
                queue.run(key+':dense-tiles',' '.join(shlex.quote(x) for x in args),budget=1800)
            dense=json.loads(target.read_text())
            if not valid_dense_semantics(dense,hashlib.sha256(source.read_bytes()).hexdigest(),
                                         hashlib.sha256(script.read_bytes()).hexdigest()):
                raise RuntimeError('18项跨列块补充语义尚未同身份通过')
        return queue.evaluate(op,source,output/'independent','codex-stride-repair',2)
    queue.job(key,verify)
    queue.state.update(status='completed',current_job=None,updated_at=nightly.now())
    nightly.save(queue.path,queue.state)
    result=queue.state['jobs'][key]
    scores=' / '.join('%.3f×'%value for value in result.get('scores',[])) or '无有效性能结果'
    (queue.batch/'进展.md').write_text('# 独立语义修复验证\n\n'
        '这是'+queue.plan.get('parent_arm','原始KG')+'生成结果之外的Codex修复，不计入原三组相同预算内的生成成绩。\n\n'
        '状态：'+result['status']+'。两次完整复验：'+scores+'。\n'+result.get('error','')+'\n')
    return result['status']=='passed'


if __name__=='__main__':
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/data/hanle/ascend-optimization'))
    parser.add_argument('--batch',required=True)
    parser.add_argument('--container',default='tle_yy')
    args=parser.parse_args()
    if not args.batch.replace('-','').replace('_','').isalnum():raise ValueError('非法批次名')
    queue=nightly.Queue(args.batch,args.root,args.container)
    print('等待本项目卡锁，主队列后验证语义修复',flush=True)
    with (args.root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            if not execute(queue):raise SystemExit(1)
        except Exception as error:
            queue.state.update(status='needs_attention',error=str(error),updated_at=nightly.now())
            nightly.save(queue.path,queue.state);raise
