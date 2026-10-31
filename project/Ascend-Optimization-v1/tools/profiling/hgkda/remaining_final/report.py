"""从原请求与确认结果生成本轮最终交付，不追加性能测量。"""
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import urllib.request

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3];ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/remaining-final-20261031'
sys.path.insert(0,str(HERE.parent/'next_phase'))
from report import verify
from knowledge import organize

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    state=read(BATCH/'next-state.json')
    if state['status']!='evidence_completed':raise ValueError('证据队列尚未完成')
    if any(v['status']=='running' for v in state['jobs'].values()):raise ValueError('仍有运行任务')
    previous=read(BATCH.parent/'remaining-nine-20261030/delivery.json');identity={};rows=[]
    for record in read(BATCH/'plan.json')['records']:
        for name,digest in record['contract_sha256'].items():
            p=BATCH/record['catalog_leaf']/name;assert sha(p)==digest;identity[str(p)]=digest
        assert sha(BATCH/record['seed'])==record['source_sha256']
    for p in BATCH.rglob('*.request.json'):
        payload=read(p)
        if payload.get('api_version')!='v6.2' or not payload.get('implementation'):continue
        result=p.with_name(p.name.replace('.request.json','.result.json'));assert result.exists(),'未知请求 '+str(p)
        identity[str(p)]=sha(p);identity[str(result)]=sha(result)
    requests=sum(p.endswith('.request.json') for p in identity)
    for p in (BATCH/'proposals').glob('*/*/manifest.json'):
        for name,digest in read(p)['files'].items():assert sha(name)==digest;identity[name]=digest
    methods=dict(fix='Ascend C向零舍入与大块事件搬运；C++分配输出时释放GIL，保留同stream任务队列和Triton布局回退',
        expm1_='Ascend C指数及近零短级数；大分块、读入缓冲写回与下一块预取，特殊布局回退Triton')
    knowledge=[]
    for op in ['fix','expm1_']:
        row=state['operators'][op];best=row['best'];confirmation=read(BATCH/'reproductions/final-confirm-01'/op/'reproduction.json')
        checked=verify(best);confirmed=verify(confirmation)
        assert checked['status']==confirmed['status']=='verified'
        assert checked['correctness_passed'] and confirmed['correctness_passed']
        assert checked['extra_status']==confirmed['extra_status']=='passed'
        assert checked['source_sha256']==confirmed['source_sha256']
        source=Path(best['source']);digest=sha(source)
        gates=[Path(best['extra_checks']),Path(best['native_safety']),Path(confirmation['extra_checks']),Path(confirmation['native_safety']),Path(row['frozen_shapes'])]
        for path in gates:
            value=read(path);assert value['source_sha256']==digest and value['status']=='passed';identity[str(path)]=sha(path)
        stable=checked['score']>=.8 and confirmed['score']>=.8
        score=math.sqrt(checked['score']*confirmed['score'])
        attempts=[]
        for p in sorted((BATCH/'proposals'/op).glob('*/assessment/formal/*.result.json')):
            value=read(p);attempts.append(dict(variant=p.parents[2].name,result=str(p),sha256=sha(p),status=value.get('status'),score=value.get('geo_mean'),passed=value.get('num_passed'),total=value.get('num_workloads')))
        diagnostic=BATCH/'diagnosis'/op/'final.json';assert read(diagnostic)['source_sha256']==digest
        out=BATCH/'deliverables'/op;out.mkdir(parents=True,exist_ok=True)
        (out/'main.py').write_bytes(source.read_bytes())
        result=dict(operator=op,stable=stable,score=score,first_score=checked['score'],confirmation_score=confirmed['score'],
            source=str(source),source_sha256=digest,backend=best['backend'],method=methods[op],
            best=best,confirmation=confirmation,checks=[dict(path=str(p),sha256=sha(p)) for p in gates],diagnostic=str(diagnostic),attempts=attempts)
        save(out/'manifest.json',result);rows.append(result)
        knowledge.append(dict(id=op+'-'+digest[:12],operator_type='pointwise',action_id='validated-implementation-'+op,
            scope=dict(operator=op,compiler='0.6.0+ascend.gitc286cba6'),summary=methods[op],status='validated',
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            evidence=dict(source=str(source),source_sha256=digest,formal_result=best['result'],formal_result_sha256=sha(best['result']),
                formal_request=checked['request'],supplementary=[dict(path=str(p),sha256=sha(p)) for p in gates])))
    paths=[]
    for p in (BATCH/'paths').glob('*/*/*/summary.json'):
        v=read(p);assert v['status']=='observed' and sha(v['trace'])==v['trace_sha256'];identity[v['trace']]=v['trace_sha256'];paths.append(str(p))
    for p in (BATCH/'actual-ir').glob('*/*/manifest.json'):
        v=read(p);assert v['entries']
        for entry in v['entries']:
            for item in entry['artifacts']:assert sha(item['path'])==item['sha256'];identity[item['path']]=item['sha256']
    assert len(paths)==12
    sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.splitlines()
    assert not [n for n in sessions if n.startswith('hanle-remaining-final-')]
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    services={};opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for port in [19655,19657,19659]:
        value=json.load(opener.open('http://127.0.0.1:%d/status'%port,timeout=15));sch=value['scheduler']
        assert sch['healthy']==1 and not any(sch[k] for k in ['active','waiting','broken']);services[str(port)]=sch
        if port==19659:assert value['software']['compiler_version']=='0.6.0+ascend.gitc286cba6' and value['software']['runtime_version']=='9.0.0'
    newly=[r['operator'] for r in rows if r['stable']];accepted=list(dict.fromkeys(previous['accepted']+newly))
    data=dict(schema='hgkda-remaining-final-delivery-v1',status='completed',time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        previous_delivery='operators/campaigns/remaining-nine-20261030/delivery.json',newly_stable=newly,accepted=accepted,
        targets=dict(total=30,regular_accepted=len(accepted),performance_pending=2-len(newly),correctness_pending=1,special=5),
        rows=rows,software_lock=read(BATCH/'software-lock.json'),timing_scope='固定KGS19659完整调用walltime；同源码两次几何平均；参考波动与候选实际延迟分开报告',
        attribution='Codex编写Ascend C与C++宿主候选、诊断和验证脚本；本批未新增CC/DeepSeek生成，不称纯原生KG自动结果')
    save(BATCH/'delivery.json',data)
    save(BATCH/'final-audit.json',dict(status='verified',formal_requests=requests,device_paths=len(paths),actual_ir_manifests=len(list((BATCH/'actual-ir').glob('*/*/manifest.json'))),
        evidence_files=identity,services=services,card_lock='available',active_batch_controllers=[],newly_stable=newly,unresolved_requests=[]))
    previous_knowledge=read(BATCH.parent/'remaining-nine-20261030/knowledge/records.json')
    save(BATCH/'knowledge/records-final.json',previous_knowledge+knowledge)
    index=organize(previous_knowledge+knowledge);assert all(x['status']=='validated' for x in index['active'] if x['id'] in {k['id'] for k in knowledge})
    save(BATCH/'knowledge/index-final.json',index)
    lines=['# 最后两项攻关交付','',f'30项目标中{len(accepted)}项常规稳定达标；本轮新增'+ '、'.join(newly)+'。另1项求逆正确性未通过，5项特殊参考条件单列。','',
        '|算子|首次完整评测|同源码确认|汇报值|完整评测|补测/新形状|','|---|---:|---:|---:|---|---|']
    for r in rows:lines.append('|'+r['operator']+'|%.3f×|%.3f×|**%.3f×**|两次30/30|原扩展、24项路径、30项新形状通过|'%(r['first_score'],r['confirmation_score'],r['score']))
    lines+=['','## 实现与结论','','- fix：'+methods['fix']+'。两次0.848×、0.875×，解决此前追加确认低于门槛的问题。',
        '- expm1_：'+methods['expm1_']+'。两次0.889×、0.868×通过完整合同；同批事件版0.687×和同步版0.710×保留。候选实际改善与参考耗时变化分开列示，不把全部分数增长解释为代码收益。',
        '- 两项均为自写Ascend C＋Triton回退，Codex完成实现与诊断；目标是0.8×验收线，当前汇报值仍低于PyTorch。',
        '- linalg_inv_ex仍原合同5/7；重复行奇异矩阵参考返回有限逆矩阵和info=0的冲突沿用前批证据，不删除输入或放宽容差。',
        '','## 方法增量','','1. 新增analyze_execution_paths.py：校验请求、源码及合同，按逐case对数贡献定位拖累项，分别计算候选/参考耗时变化，结合设备路径提出下一项实验。',
        '2. 复用成对profiling、真实IR、经验Roofline及热点工具；本轮增加后端与分块/缓冲/宿主路径的实现验证，没有新增RTL或完整关键路径重建。',
        '3. 快速路径24项、冻结后新形状30项和一次固定确认进入交付门禁；通过源码/请求/结果身份约束更新分类知识。',
        '4. 本批复杂机制与实现选择由Codex完成，脚本计算事实并执行门禁；未把它描述为全部自动发现的Agent。',
        '',f'已核验{requests}份正式请求、12条设备路径、2份实际Triton IR。Windows同步完成情况另见同步清单。共享服务保持健康空闲，本批控制器退出、卡锁释放。']
    lines+=['','## 候选与参考耗时变化','','下表比较选中版与本批event-large-v6；比值小于1表示耗时下降。这是两个评测窗口的实际记录，不能仅凭比值证明某项代码修改的因果收益。','','|算子|候选耗时比|参考耗时比|候选实际加速|','|---|---:|---:|---:|']
    for r in rows:
        drift=read(r['diagnostic'])['drift']
        lines.append('|'+r['operator']+'|%.4f|%.4f|%.4f×|'%(drift['candidate_time_ratio'],drift['reference_time_ratio'],drift['candidate_actual_speedup']))
    (BATCH/'交付报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (BATCH/'进展.md').write_text('# 本批已完成\n\n本页已由最终审计更新；当前累计24/30常规稳定达标。fix和expm1_的参考路径、设备trace、实际IR和新形状检查均已收尾。详细分数、所有尝试、源码与证据身份见[交付报告](交付报告.md)及delivery.json。\n',encoding='utf8')
    (BATCH/'接续说明.md').write_text('# 接续说明\n\n本批已完成，权威索引为delivery.json；累计24/30常规达标，linalg_inv_ex和5项特殊参考单列。\n\n入口：tools/profiling/hgkda/remaining_final/README.md。复现用remaining/reproduce.py --batch remaining-final-20261031，指定算子及新的run-name，不覆盖原始结果。\n\n源码为自写Ascend C＋Triton回退，保留二进制与构建身份。fix采用queue-gil-v12，expm1_采用prefetch-large-v8。参考波动和实际候选延迟分开，正式walltime与独立trace分开。未自动提交GitHub或修改语雀。\n',encoding='utf8')
    print(json.dumps(dict(status='verified',newly_stable=newly,targets=data['targets'],formal_requests=requests,device_paths=len(paths))))

if __name__=='__main__':main()
