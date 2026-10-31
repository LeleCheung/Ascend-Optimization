"""从原始请求、源码、正式结果及门禁生成中文交付；不把运行状态当成果。"""
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'
sys.path.insert(0,str(HERE.parent/'next_phase'))
from report import verify
from validation_status import supplement_status


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build():
    state=read(BATCH/'next-state.json');rows=[];accepted=[]
    methods={
        'negative_':'连续对齐输入用Ascend C取负，双缓冲队列与C++启动；保留原地及有符号零',
        'log2_':'Ascend C计算Ln及常量乘法，较大流式分块与C++启动',
        'fix':'Ascend C直接向零舍入；Python分配新输出，C++启动',
        'round_out':'Ascend C直接向最近偶数舍入，双缓冲搬运；小数位和布局回退Triton',
        'special_round_out':'Ascend C直接向最近偶数舍入，较大分块与C++启动；小数位和布局回退Triton',
        'hardswish':'保留原融合公式，改用Ascend C双缓冲与C++启动，Python分配独立输出',
        'fmod_tensor_':'Ascend C除法、截断、乘减计算，双缓冲与C++启动；广播和stride保留回退',
        'expm1_':'Ascend C指数运算及小值补偿，双缓冲；当前性能未达标',
        'linalg_inv_ex':'检查LU除法、倒数缩放与FMA顺序；原合同尚未通过'}
    for op in read(BATCH/'plan.json')['priority_operators']:
        row=state.get('operators',{}).get(op,{})
        start=row.get('starting_point') or {};best=row.get('best') or {};confirm=row.get('final_confirmation') or {}
        checked=verify(best);confirmation=verify(confirm)
        additional=[verify(item) for item in row.get('additional_confirmations',[])]
        native=read(best['native_safety']) if best.get('native_safety') else None
        native_ok=not best.get('native_manifest') or (native and native.get('status')=='passed' and native['source_sha256']==best['source_sha256'])
        stable=(checked.get('status')=='verified' and checked.get('correctness_passed') and checked.get('extra_status')=='passed'
            and checked.get('score',0)>.8 and native_ok and confirmation.get('status')=='verified'
            and confirmation.get('correctness_passed') and confirmation.get('score',0)>.8 and confirmation.get('source_sha256')==checked.get('source_sha256')
            and all(item.get('status')=='verified' and item.get('correctness_passed') and item.get('score',0)>.8 and item.get('source_sha256')==checked.get('source_sha256') for item in additional))
        valid_scores=[item['score'] for item in [checked,confirmation,*additional] if item.get('status')=='verified' and item.get('correctness_passed') and isinstance(item.get('score'),(int,float))]
        score=math.exp(sum(math.log(v) for v in valid_scores)/len(valid_scores)) if valid_scores else None
        status='新增稳定达标' if stable else '完整正确，性能未达标' if checked.get('correctness_passed') and checked.get('extra_status')=='passed' else '正确性未通过'
        if additional and not stable and checked.get('correctness_passed'):
            status='正确性通过，追加确认低于0.8×，未稳定达标'
        if stable:accepted.append(op)
        rows.append(dict(operator=op,starting=start,best=best,verified=checked,confirmation=confirmation,
                         additional_confirmations=additional,native_safety=native,status=status,stable=bool(stable),score=score,backend=best.get('backend','Triton'),method=methods[op]))
    previous=read(BATCH.parent/'threeway-20261030/delivery.json')
    data=dict(schema='hgkda-remaining-nine-delivery-v1',status='completed' if state['status']=='all_completed' else 'in_progress',
        time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        newly_stable=accepted,accepted=[*previous['accepted'],*accepted],previous_delivery='operators/campaigns/threeway-20261030/delivery.json',
        threshold=.8,timing_scope='固定KGS19659 walltime完整调用；两次同源码有效完整结果取几何平均，低分保留',
        rows=rows,software_lock=read(BATCH/'software-lock.json'),
        attribution='Codex诊断、源码实现与流程编排；实际KG搜索另有CC/DeepSeek工作区证据；不称为纯原生KG或纯Triton全部成果')
    data['targets']=dict(total=30,regular_accepted=len(data['accepted']),
        performance_pending=sum(not row['stable'] for row in rows if row['operator']!='linalg_inv_ex'),
        correctness_pending=sum(not row['stable'] for row in rows if row['operator']=='linalg_inv_ex'),special=5)
    data['sources']=dict(previous['sources'])
    for row in rows:
        if not row['stable']:continue
        best=row['best'];source=Path(best['source'])
        data['sources'][row['operator']]=dict(source=source.relative_to(PROJECT).as_posix(),source_sha256=best['source_sha256'],
            score=row['score'],complete=[best['num_passed'],best['num_workloads']],backend=row['backend'],
            supplementary=best['extra_checks'],native_safety=best.get('native_safety'),
            results=[best['result'],row['confirmation']['result'],*[item['result'] for item in row['additional_confirmations']]])
    (BATCH/'delivery.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['# 剩余九项攻关交付','','本批相对PyTorch完整调用加速比；超过0.8×为达标，超过1×才比PyTorch快。', '',
        '|算子|本批历史优化起点|HG-KDA|完整评测|状态|','|---|---:|---:|---|---|']
    for row in rows:
        start=row['starting'];best=row['best']
        tested=best or start
        start_text='%.3f×'%start['geo_mean'] if start.get('status')=='PASSED' and start.get('geo_mean') else str(start.get('num_passed','—'))+'/'+str(start.get('num_workloads','—'))
        score='%.3f×'%row['score'] if row['score'] else '—'
        lines.append('|`'+row['operator']+'`|'+start_text+'|'+score+'|'+str(tested.get('num_passed','—'))+'/'+str(tested.get('num_workloads','—'))+'|'+row['status']+'|')
    lines+=['', '本批新增稳定达标 '+str(len(accepted))+' 项；加上前批16项，当前常规达标共 '+str(len(data['accepted']))+' 项。原30项目标中的条件验收、参考CPU回退问题继续单列，不能计为常规NPU达标。', '',
        '源码、原请求、完整结果、扩展补测、原生对齐路径补测和确认结果见 delivery.json。Ascend C来源可见，特殊布局回退Triton；并非使用PyTorch目标函数替代候选。fix第二次接近门槛，因此预先固定再补一次，三次全部保留并取几何平均，不挑最好一次。', '',
        '成对设备trace与实际IR位于 paths/、actual-ir/。独立profiling耗时不替代正式加速比。失败提案、数值路径差异及未达标实现保留。', '',
        '## 实际改法', '',
        '本轮主要将已验证的启动和分块方法迁移到来源可见的Ascend C实现，特殊布局保留Triton。多数算子的起点已是单kernel融合，不能把本轮提升统一归因为新增融合。输出地址、当前stream和动态元数据逐次读取，缓存中不保存张量值。', '']
    for row in rows:lines.append('- `'+row['operator']+'`：'+row['method']+'。')
    lines+=['', '## 未完成项', '',
        '`fix` 三次完整正确性都通过，加速比分别约0.892×、0.803×、0.766×，几何平均0.819×；第三次低于门槛，不算稳定达标，不继续按结果反复重测挑高分。', '',
        '`expm1_` 起点与PyTorch同输入的真机成对采集已完成。最大FP32输入搬运量相同，但起点MTE2指令数为参考约8.56倍、Vector报告时间约5.85倍；这是当前起点的设备执行差距，不能冒称最终Ascend C候选的同一项因果证明。新候选虽改善，仍需继续降低大输入时间。原件见 analysis/expm1_/。', '',
        '`linalg_inv_ex` 的冻结3×3输入存在完全重复行。精确有理数核验表明数学上不可逆，而本机参考返回info=0及约1.05亿量级的有限逆矩阵，A×inverse-I最大残差约3.30。候选判断奇异会与现有逐值参考冲突；本轮保留原输入、容差、失败结果，不计达标。具体证据见 analysis/linalg_inv_ex/reference-mathematical-conflict.json。', '',
        '本轮源码实现与机制诊断有Codex实质参与，CC/DeepSeek的KG搜索及结果另存 results/*/search-1/。两者贡献分别记账。', '',
        '## 复现', '',
        '在910B宿主机项目目录执行，使用新的run-name保存新结果，不替换本批原始成绩：', '',
        '```bash',
        'cd /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1',
        '/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python tools/profiling/hgkda/remaining/reproduce.py --operator fmod_tensor_ --run-name manual-check-01',
        '```', '',
        '完整工具和构建说明见 [remaining/README.md](../../../tools/profiling/hgkda/remaining/README.md)。复现程序持有同一物理卡7锁，执行完整KGS、扩展补测和原生路径检查；源码与构建附件按SHA核验。']
    (BATCH/'交付报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status=data['status'],newly_stable=accepted,total=len(data['accepted'])),ensure_ascii=False))
    return data


if __name__=='__main__':build()
