#!/usr/bin/env python3
"""五算子队列及补验退出后，串行修复scatter跨目标前缀相消。"""
import argparse
import datetime
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import time

TOOLS=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('scatter_followup_base',TOOLS/'run-endurance.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
MAIN='remaining-five-20261023-v2'
SEED_SHA='8a67a004fd75f9332094bcc3cc821e9d2dfa31dbcc10b474ac27937da9c1a04f'
REPO=base.ROOT/'Ascend-Optimization'
PROJECT=REPO/'project/Ascend-Optimization-v1'


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def prepare(name):
    batch=PROJECT/'operators/campaigns'/name
    if batch.exists():raise FileExistsError(batch)
    parent=PROJECT/'operators/campaigns'/MAIN
    plan=base.read(parent/'plan.json')
    record=dict(next(r for r in plan['records'] if r['operator']=='scatter_add'))
    state=base.read(parent/'endurance-state.json')
    seed=Path(state['operators']['scatter_add']['best']['source'])
    if base.sha(seed)!=SEED_SHA:raise ValueError('待修复源码身份不同，不能静默替换起点')
    leaf=batch/record['catalog_leaf'];leaf.mkdir(parents=True)
    for filename,digest in record['contract_sha256'].items():
        origin=parent/record['catalog_leaf']/filename
        if base.sha(origin)!=digest:raise ValueError('原合同变化')
        shutil.copyfile(origin,leaf/filename)
    target=batch/'seeds/scatter_add.py';target.parent.mkdir();shutil.copyfile(seed,target)
    record.update(seed='seeds/scatter_add.py',source_sha256=SEED_SHA,
                  seed_origin=str(seed),seed_scope='前一批随机数值门禁通过，但跨目标大数/小数反例待修复')
    catalog=base.read(parent/'catalog/manifest.json');catalog['name']=name
    save(batch/'catalog/manifest.json',catalog)
    knowledge=batch/'knowledge';knowledge.mkdir()
    for filename in ['散射排序后的分段扫描.md','散射首轮机制证据与第二轮方向.md',
                     'scatter-first-pass-validated.json','scatter-range-cpu-counterexample.json']:
        shutil.copyfile(parent/'knowledge'/filename,knowledge/filename)
    brief='''# scatter_add分段归约正确性与性能修复

目标：修复不同target之间的前缀相消，并保持完整KGS调用超过0.8×。起点8a67a004虽在上一批正式18、补充18、全部timing9通过且1.306×，但其全行FP32 prefix[end]-prefix[begin]会丢失另一个target的小值，不得当作最终有效交付。

反例：input全0，src前部向target0累加大数，最后一个src=.25向target1。target1的正确结果独立于target0，必须为.25。CPU算法模型18项全部暴露此问题；本批先对原候选跑NPU检查。冻结合同与容差保持原样，新增check-scatter-dynamic-range.py仅作非计分门禁。

优先替换为按target分段的FP32扫描：排序后key可解码target，tuple associative_scan的combine为(key_a,sum_a),(key_b,sum_b) -> (key_b,where(key_a==key_b,sum_a+sum_b,sum_b))。输出读取本target段尾累加值，不再减去其他target的大前缀。保留片上二分和移位/掩码，不重复GM随机读。tuple扫描能否本机编译、缓冲与速度必须实际验证，必要时用分组/两阶段方式，不允许改用被优化的PyTorch/FlagGems算子。

注意标准的分段扫描combine应保留分段重置信息；可以采用(sum,head_flag)与OR传播head_flag的关联组合，或证明有序key相等判定组合在本次有序序列上成立。必须真实验证，不假定任意不关联函数可以并行扫描。

保持非末轴、不同stride、src和index范围语义、低精度FP32累加、空输入及当前stream。18项原补充、9项timing数值、18项动态范围检查必须全部通过，再作一次完整KGS正式复验；按同case候选实际延迟选版。不得缩小排序范围到未覆盖测试外以绕过数值问题。可以合理保留通用原子路径，但不能用数据/随机种子/用例编号硬编码。

只用物理卡7和KGS19659，串行卡锁；不升级环境或重启共享服务。原生完整调用PyTorch参考，不替换参考或放宽容差。Codex提供反例、机制建议与门禁，CC/DeepSeek实际生成及工具调用分别登记。两轮搜索均继续到预算或有效交付，不阶段性询问用户。
'''
    (knowledge/'夜间任务.md').write_text(brief)
    names=['run-scatter-segmented-followup.py','check-scatter-dynamic-range.py','run-endurance.py',
           'check-endurance.py','observe-endurance.py','run-nightly.py','build-kg-overlay.py','audit-endurance.py']
    save(knowledge/'tool-sha256.json',{n:base.sha(TOOLS/n) for n in names})
    plan.update(schema_version='hgkda-scatter-segmented-followup-v1',records=[record],duration_hours=3,
                passes=[dict(rounds=4,budget_seconds=3000),dict(rounds=3,budget_seconds=2400)],
                parent_campaign=MAIN,policy='新增非计分数值门禁；不改原始正式合同；主批次和补验退出后才持卡执行')
    save(batch/'plan.json',plan)
    (batch/'README.md').write_text('# scatter_add跨目标相消修复\n\n这是'+MAIN+'的继续修复批次，不覆盖历史结果。先等待主队列及补验退出，再持同一物理卡7锁执行。原始正式合同不变，额外要求大数/小数分属不同target时互不干扰。任务、方法预算、贡献及复现入口见knowledge/夜间任务.md和本控制器。最终看deliverables/manifest.json与completion-audit.json，运行中不算交付。\n')
    return batch


class Campaign(base.Campaign):
    def gates(self,op,source,out):
        previous=super().gates(op,source,out)
        if previous['status']!='passed':return previous
        target=out/'dynamic-range.json'
        if not target.exists():
            try:self.run('dynamic-range-'+out.name,[base.PY,TOOLS/'check-scatter-dynamic-range.py',
                         '--source',source,'--output',target],1800)
            except RuntimeError:
                if not target.exists():raise
        data=base.read(target)
        if data['source_sha256']!=base.sha(source):raise ValueError('动态范围源码不同')
        if data['status']!='passed':return dict(status='semantic_failed',check=str(target),passed=data['passed'],total=data['total'])
        return dict(status='passed',checks=previous['checks']+[str(target)])


def audit(batch):
    module_spec=importlib.util.spec_from_file_location('scatter_audit',TOOLS/'audit-endurance.py')
    helper=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(helper)
    state=base.read(batch/'endurance-state.json');plan=base.read(batch/'plan.json')
    if state['status']!='completed':raise ValueError('修复队列未结束')
    record=plan['records'][0];tools=base.read(batch/'knowledge/tool-sha256.json')
    for name,digest in tools.items():
        if base.sha(TOOLS/name)!=digest:raise ValueError('冻结工具字节不同')
    leaf=batch/record['catalog_leaf']
    for filename,digest in record['contract_sha256'].items():
        if base.sha(leaf/filename)!=digest:raise ValueError('合同身份改变')
    manifest=base.read(batch/'deliverables/manifest.json')
    prefix='/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/'+batch.name+'/'
    def local(path):
        if not str(path).startswith(prefix):raise ValueError('证据不在本批次')
        value=batch/str(path)[len(prefix):];value.resolve().relative_to(batch.resolve());return value
    if not manifest['records']:
        return dict(status='unclosed',delivered=0,reason='没有通过所有门禁的源码，保留失败')
    if len(manifest['records'])!=1:raise ValueError('交付对象不同')
    selected=manifest['records'][0];digest=selected['selected']['source_sha256']
    if base.sha(local(selected['delivered']))!=digest:raise ValueError('交付源码不同')
    pool=[]
    for key,job in state['jobs'].items():
        if key!='parent:scatter_add' and not key.startswith('validate:scatter_add:'):continue
        value=job.get('result') or {}
        if job.get('status')!='completed' or value.get('status')!='passed':continue
        source=local(value['source']);sha=base.sha(source)
        if sha!=value['source_sha256']:raise ValueError('候选源码改变')
        checks=[base.read(local(p)) for p in value['gates']]
        if len(checks)!=2:raise ValueError('门禁缺失')
        for check in checks:
            if (check['source_sha256']!=sha or check['status']!='passed' or check['passed']!=check['total']
                    or check['total']!=18 or check['physical_device']!=7 or len(check['cases'])!=18
                    or any(r['status']!='passed' for r in check['cases'])):
                raise ValueError('补充门禁不通过')
        normal,dynamic=checks
        if normal['checker_sha256']!=tools['check-endurance.py'] or dynamic['checker_sha256']!=tools['check-scatter-dynamic-range.py']:
            raise ValueError('门禁工具不同')
        timing=normal['timing_input_correctness']
        timing_names=[json.loads(line)['name'] for line in (leaf/'timing.jsonl').read_text().splitlines() if line.strip()]
        if (not (timing['passed']==timing['total']==9) or [r['case'] for r in timing['cases']]!=timing_names
                or any(r['status']!='passed' or r.get('repetitions')!=2 or not r.get('stream_switch_checked') for r in normal['cases']+timing['cases'])
                or timing['contract_sha256']!={n:record['contract_sha256'][n] for n in ['oracle.py','timing.jsonl']}):
            raise ValueError('性能输入数值或补充覆盖缺失')
        expected={(dtype,width,sign) for dtype in ['torch.float16','torch.bfloat16','torch.float32'] for width in [257,1024,4096] for sign in [1,-1]}
        if {(r['dtype'],r['width'],r['sign']) for r in dynamic['cases']}!=expected or any(r['status']!='passed' or r.get('repetitions')!=2 or not r.get('stream_switch') for r in dynamic['cases']):
            raise ValueError('动态范围逐项证据缺失')
        path=local(value['result']);result=base.read(path)
        helper.verify_formal_contract(record,path.parent,result)
        if result['status']!='PASSED' or not (result['num_passed']==result['num_workloads']==18):raise ValueError('正式评测失败')
        if value['geo_mean']!=result['geo_mean']:raise ValueError('候选成绩与正式结果不同')
        request=base.read(path.with_name(path.name.replace('.result.json','.request.json')))
        sources=request['implementation']['sources']
        if len(sources)!=1 or sources[0]['content']!=source.read_text():raise ValueError('正式请求源码不同')
        pool.append(dict(source_sha256=sha,result=result))
    names=[json.loads(line)['name'] for line in (leaf/'timing.jsonl').read_text().splitlines() if line.strip()]
    choice=helper.verify_fastest(digest,pool,names)
    return dict(status='passed',delivered=1,source_sha256=digest,score=selected['selected']['geo_mean'],
                meets_target=selected['selected']['geo_mean']>.8,selection=choice,
                formal=18,supplementary=18,timing_numerical=9,dynamic_range=18)


def execute(name):
    batch=PROJECT/'operators/campaigns'/name
    if not batch.exists():raise FileNotFoundError('先用--prepare生成可审阅的批次')
    main=PROJECT/'operators/campaigns'/MAIN
    deadline=time.monotonic()+24*3600
    waiter=batch/'wait-state.json'
    with (base.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        while True:
            state=base.read(main/'endurance-state.json')
            post=base.read(main/'postflight-state.json') if (main/'postflight-state.json').exists() else {}
            if state['status']=='completed' and post.get('status') in ['passed','needs_attention']:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:pass
            save(waiter,dict(status='waiting_for_main_and_postflight',main=state['status'],postflight=post.get('status'),time=datetime.datetime.utcnow().isoformat()+'Z'))
            if time.monotonic()>deadline:raise TimeoutError('等待预算结束，未用卡')
            time.sleep(30)
        save(waiter,dict(status='running'))
        campaign=Campaign(name)
        try:
            campaign.execute()
            result=audit(batch);save(batch/'completion-audit.json',result)
            save(waiter,dict(status='passed' if result['status']=='passed' and result.get('meets_target') else 'needs_attention',audit=result))
        except Exception as error:
            campaign.state.update(status='needs_attention',error=type(error).__name__+': '+str(error));campaign.save()
            save(waiter,dict(status='needs_attention',error=str(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',required=True)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:print(prepare(args.batch))
    else:execute(args.batch)
