#!/usr/bin/env python3
"""将夜间交付汇总为两张中文简表，独立记录新增源码和真实耗时收益。"""
import argparse
import json
from pathlib import Path


def summarize(batch):
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    state=read(batch/'endurance-state.json');delivery=read(batch/'deliverables/manifest.json')
    if state['status']!='completed':raise ValueError('批次未结束')
    plan=read(batch/'plan.json');selected={r['operator']:r for r in delivery['records']}
    targets={r['operator']:r for r in read(Path(__file__).with_name('targets.json'))['targets']}
    lines=['# 夜间连续优化精简汇报','','本批固定FlagGems d6a8eec、FlagGems-SGLang 9386120和FlagTree 0.6.0+ascend.gitc286cba6，使用物理卡7、KGS完整调用walltime。新源码一次独立完整复验；按同case候选实际延迟选版。','']
    fmt=lambda x:'%.3f×'%x if isinstance(x,(int,float)) else '—'
    facts=[]
    for group,label in [('excel_meeting','会议指定算子'),('flaggems_sglang','FlagGems-SGLang')]:
        lines+=['## '+label,'','| 算子 | FlagGems历史实测 | 同批起点 | HG-KDA当前版 | 相对起点实际加速 | 结果 |','|---|---:|---:|---:|---:|---|']
        for record in plan['records']:
            if record['source']!=group:continue
            op=record['operator'];d=state['operators'].get(op,{});row=selected.get(op)
            target=targets[op];history=target.get('latest_baseline_check',{})
            historical=(fmt(history.get('relative_pytorch')) if history.get('status')=='PASSED' else history.get('status','—'))
            parent=d.get('parent') or {}
            parent_score=fmt(parent.get('geo_mean')) if parent.get('status')=='passed' else parent.get('status','未完成')
            if row:
                best=row['selected'];score=fmt(best['geo_mean']);speedup=fmt(row.get('actual_speedup_over_parent'))
                message='新源码，完整及补充检查通过' if row['new_source'] else '保留起点，没有新增收益'
                if best['geo_mean']<.8:message+='；未达0.8×'
            else:score=speedup='—';message='无通过全部门禁的交付'
            lines.append('| `'+op+'` | '+historical+' | '+parent_score+' | '+score+' | '+speedup+' | '+message+' |')
            facts.append(dict(operator=op,source_group=group,parent=parent,selected=row,status=d.get('status'),
                attempts=d.get('attempts',[])))
        lines.append('')
    origin=('本批是'+plan['parent_batch']+'之后的独立追加修复，起点为父批次选中的有效CC源码，不是原版FlagGems；新增源码由Codex机械修复采集兼容和启动语义，保持计算kernel。'
            if plan.get('parent_batch') else 'SELU起点为上批优化源码，其余起点为固定FlagGems源码。')
    lines+=['FlagGems历史值来自已归档固定版本，不与本批参考波动混算。'+origin+'相对起点收益按候选实际耗时计算。正确性失败时不列整组加速比。','',
        'Codex负责诊断先验、连续控制和追加排障；CC/DeepSeek负责源码生成和原生KG工具调用。真实载入角色、提示词、模型响应、源码和逐case原件保留。模拟器超时或取消单列，不称为成功指令热点分析；本批没有建立定量Roofline。','',
        '最终源码见`deliverables/manifest.json`，逐轮生成与独立门禁见`results/`，真机任务路径、IR和metrics见`research/`。实际失败与恢复见`endurance-state.json`和修正回执。']
    audit=batch/'completion-audit.json'
    if audit.exists():
        a=read(audit);lines+=['','离线核验：'+a['status']+'，'+str(a['checks'])+'项；交付'+str(a['delivered'])+'/'+str(a['registered'])+'个登记对象。未闭环对象：'+('、'.join(a['unclosed']) or '无')+'。']
        if a.get('profiling_gaps'):lines+=['','最终采集尚未完整完成的对象：'+'、'.join(a['profiling_gaps'])+'。源码通过核验不代表采集完整，须结合原始失败记录与追加修复批次。']
    (batch/'精简汇报.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (batch/'final-comparison.json').write_text(json.dumps(dict(status='completed',records=facts),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(records=len(facts),delivered=len(selected)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-dir',type=Path,required=True)
    a=p.parse_args();summarize(a.batch_dir)
