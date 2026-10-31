"""从已核验原件生成简报及逐case性能表，不修改最佳选择。"""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from report import build


def timing(record):
    value=json.loads(Path(record['result']).read_text(encoding='utf8'))
    return {row['uuid']:row for row in value['per_workload'] if row['phase']=='timing'}


def main():
    project=Path(__file__).resolve().parents[4]
    batch=project/'operators/campaigns/threeway-20261030'
    index=build(batch)
    state=json.loads((batch/'next-state.json').read_text(encoding='utf8'))
    score=lambda row:'%.3f×'%row.get('display_score',row.get('score',0))
    counts=lambda row:'/'.join(str(n) for n in row.get('complete',[]))
    qualified=lambda row:row.get('status')=='qualified' and row.get('score',0)>.8
    accepted=[];new=[];facts={};csv_rows=[];seen_results=set();threeway=[]
    for row in index['rows']:
        best=row['best'];initial=row['initial'];op=row['operator']
        changed=best.get('source_sha256')!=initial.get('source_sha256')
        stable=(qualified(best) and best.get('stable_above_threshold',False)
                and (not changed or qualified(best.get('confirmation',{}))))
        if op=='cross_out' and changed:
            stable=stable and bool(best.get('source_record',{}).get('cross_safety'))
        if stable:accepted.append(op)
        if stable and changed:new.append(op)
        records=[('本批起点',initial),('当前最佳首次完整评测',best),('CANN参考',row['reference']),('AscendC初版',row['native_v1'])]
        if best.get('confirmation'):records.append(('当前最佳完整确认',best['confirmation']))
        records += [('同源码独立完整测量',entry) for entry in best.get('independent_records',[])]
        for label,entry in records:
            if entry.get('status')!='qualified':continue
            if entry['result'] in seen_results:continue
            seen_results.add(entry['result'])
            for case,value in timing(entry['source_record']).items():
                csv_rows.append(dict(算子=op,用例=case,实现=label,完整调用微秒=value['latency_ms']*1000,
                    配对PyTorch微秒=value['reference_latency_ms']*1000,
                    相对PyTorch=value['reference_latency_ms']/value['latency_ms'],
                    源码SHA=entry['source_sha256'],原始结果=entry['result']))
        if changed and best.get('status')=='qualified':
            old=timing(initial['source_record']);versions=[timing(entry['source_record']) for entry in best['independent_records']]
            final_latency={key:math.exp(sum(math.log(version[key]['latency_ms']) for version in versions)/len(versions)) for key in old}
            final_reference={key:math.exp(sum(math.log(version[key]['reference_latency_ms']) for version in versions)/len(versions)) for key in old}
            facts[op]=dict(actual_candidate_latency_gain=math.exp(sum(math.log(old[key]['latency_ms']/final_latency[key]) for key in old)/len(old)),
                paired_reference_latency_change=math.exp(sum(math.log(final_reference[key]/old[key]['reference_latency_ms']) for key in old)/len(old)),
                interpretation='实际候选延迟改善与配对参考漂移分别计算，不能将汇总加速比之比全部当作代码收益')
    by_op={row['operator']:row for row in index['rows']}
    norm=by_op['native_group_norm']
    def kernel_identity(relative):
        source=(project/relative).read_text(encoding='utf8')
        tree=ast.parse(source)
        node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_gn_kernel')
        return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
    facts['native_group_norm_kernel_source_unchanged']=(kernel_identity(norm['initial']['source'])==kernel_identity(norm['best']['source']))
    old=batch/'native-candidates/log10_/ascendc-ln-v1/libhgkda_log10.so'
    new_binary=batch/'proposals/log10_/ascendc-taskqueue-v2/libhgkda_log10.so'
    facts['log10_ascendc_v1_v2_device_binary_unchanged']=hashlib.sha256(old.read_bytes()).hexdigest()==hashlib.sha256(new_binary.read_bytes()).hexdigest()
    lines=['# 三路对照与算子优化交付','','截至北京时间'+index['time_beijing']+'；执行状态：'+str(index['status'])+'。',
        '',f'本轮16项源码中，{len(accepted)}项通过完整合同与补测并超过0.8×；12项历史达标源码已原样复验。新增或恢复稳定达标：'+('、'.join(new) or '暂无')+'。',
        '', '## 四项重点结果','', '|算子|本批起点|当前最佳|完整评测|实现与状态|','|---|---:|---:|---|---|']
    for op in index['priority_operators']:
        row=by_op[op];best=row['best']
        status='两次完整评测稳定过线' if op in new else '本批复验达标' if op in accepted else '正确，性能待优化'
        lines.append('|'+op+'|'+score(row['initial'])+'|'+score(best)+'|'+counts(best)+'|'+best.get('backend','—')+'；'+status+'|')
    lines+=['','数值均为相对同环境PyTorch的KGS完整调用加速比。同一源码已有多份独立完整有效结果时，简表汇总全部记录的几何平均，保留低于0.8×的测量；稳定过线要求这些测量均过线。新增过线源码追加一次同条件完整确认，其余默认一次完整评测。所有逐case原始耗时及配对PyTorch耗时见[逐case性能表](逐case性能.csv)。',
            '', '## 有效修改','',
            '- `log10_`：连续对齐输入使用源码自建Ascend C，其他布局保留Triton。将Python元数据处理、ctypes及逐调用宿主队列刷新改为C++读取实时元数据并排入同一stream任务队列；不缓存输入值或旧地址。Ascend C初版与任务队列版的设备二进制SHA完全一致。',
            '- `native_group_norm`：保留原Triton计算，缓存连续布局的分派和启动常量，使用`new_empty`新建三个独立输出，保留非连续布局路径。核对AST确认`_gn_kernel`源码未变。',
            '- `cross_out`：用移位的连续读取代替大块动态gather，块内三分量位置用已核验有限索引范围内的浮点倒数及截断计算，避开慢整数取模；大输入使用4096分块。交付版每次复核dim/布局，保留storage别名与真实读写边界。',
            '- `nan_to_num_`：默认边界在编译时确定，减少启动运行时参数和Python分派；小中输入缩短宿主路径，大输入每个program依次处理多个分块以摊薄初始化成本。非连续、自定义替换值和特殊值保留通用路径。',
            '', '## CANN原生库参考','', '|算子|本批Triton起点|PyTorch|CANN参考|参考类型|完整评测|','|---|---:|---:|---:|---|---|']
    for op in ['log10_','hardswish_out','native_group_norm']:
        row=by_op[op];reference=row['reference']
        lines.append('|'+op+'|'+score(row['initial'])+'|1.000×|'+score(reference)+'|'+reference.get('backend','—')+'|'+counts(reference)+'|')
        candidate=row['best'];versions=[timing(entry['source_record']) for entry in candidate['independent_records']]
        native=timing(reference['source_record']);keys=set(native)
        assert all(set(version)==keys for version in versions)
        candidate_time={key:math.exp(sum(math.log(version[key]['latency_ms']) for version in versions)/len(versions)) for key in keys}
        reference_time={key:math.exp(sum(math.log(version[key]['reference_latency_ms']) for version in versions)/len(versions)) for key in keys}
        geo=lambda values:math.exp(sum(math.log(value) for value in values)/len(values))
        threeway.append(dict(算子=op,HGKDA完整调用微秒=geo(list(candidate_time.values()))*1000,
            配对PyTorch完整调用微秒=geo(list(reference_time.values()))*1000,
            CANN完整调用微秒=geo([native[key]['latency_ms'] for key in keys])*1000,
            HGKDA相对PyTorch=candidate['display_score'],
            HGKDA相对CANN=geo([native[key]['latency_ms']/candidate_time[key] for key in keys]),
            HGKDA实现=candidate['backend'],CANN实现=reference['backend'],
            口径='同输入逐case实际耗时几何汇总；CANN与候选属同批独立评测窗口，不由汇总speedup相除推导'))
    lines+=['','CANN列包含实际宿主包装成本，源码与二进制身份、原请求和补测均保留；PyTorch可能调用相同aclnn内核。源码自建Ascend C候选与厂商CANN库参考分别登记。三路实际完整调用耗时及逐case推导的相对CANN性能见[三路对照表](三路对照.csv)。',
        '', '## 机制与失败记录','',
        '实际候选调用耗时变化与配对参考漂移分别保存在[机制事实](机制事实.json)。`log10_`同一设备二进制的初版／任务队列版，在代表输入中设备任务耗时接近，宿主提交明显缩短。完整调用收益仍以KGS为准；同进程的逐次同步时间不替换正式成绩。',
        '', '`cross_out`扩大分块、直接stride3读写、二维split/join均未改善整体性能；KG后续采用去整数取模、移位连续读取及更大分块得到新候选。原高分源码的尾块读取、地址memo和共享storage别名问题分别保留，修复源码必须重新验收。块内浮点倒数替代整数除法仅在当前最大4096索引范围内穷举核验，不推广为无界整数等价式。`nan_to_num_`固定40个program的流式循环更慢，保留原始结果及较快实现。',
        '', 'KG的`cross_out`第一轮msprof多次采集未产生有效完成标记，指标和指令附件缺失；其分析明确记录dominant_bound=unknown。不能将后续提升归因于已实测的MTE/Vector瓶颈。改法来自源码、完整延迟及独立小实验；最终源码另采设备trace和实际IR。KG/CC续搜结果与Codex诊断、候选及语义修复分别记账。',
        '', '## 目标账本','', '|目标组|常规达标|性能待优化|正确性待解决|特殊条件单列|合计|','|---|---:|---:|---:|---:|---:|',
        f'|1015|{6+int("log10_" in new)}|{1-int("log10_" in new)}|0|3|10|',
        f'|1029去重新增|{6+sum(op in new for op in ["cross_out","nan_to_num_","native_group_norm"])}|{11-sum(op in new for op in ["cross_out","nan_to_num_","native_group_norm"])}|1|2|20|',
        '', '特殊条件继续保留：累计积分的官方确定性条件；池化、共轭、量化缓存掩码和带更新BatchNorm的参考CPU回退。`linalg_inv_ex`仍是正确性问题。本轮不改写这些历史结论。',
        '', '## 复现','', '在910B宿主项目目录执行，每次使用新run-name：','',
        '```bash',
        '/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python tools/profiling/hgkda/threeway/reproduce.py --operator log10_ --run-name manual-20261031',
        '/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python tools/profiling/hgkda/threeway/reproduce.py --operator native_group_norm --run-name manual-20261031',
        '```','', '卡锁、固定合同、完整评测和独立补测由脚本处理。构建、接续及已修复问题见[工具说明](../../../../tools/profiling/hgkda/threeway/README.md)；所有源码、逐case结果及附件身份见[已核验进展](已核验进展.md)。',
        '', '固定环境：FlagGems `d6a8eec473517a3d68157b208eb9c057eb1d4c50`；FlagTree `0.6.0+ascend.gitc286cba6`；CANN 9.0.0；驱动25.2.0；Torch 2.9.0／torch_npu 2.9.0.post2；物理卡7。']
    (batch/'交付说明.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (batch/'机制事实.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (batch/'逐case性能.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    with (batch/'三路对照.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(threeway[0]));writer.writeheader();writer.writerows(threeway)
    delivery=dict(status='completed' if state['status']=='all_completed' else 'running',
        time_beijing=index['time_beijing'],accepted=accepted,newly_stable=new,
        sources={row['operator']:dict(source=row['best']['source'],source_sha256=row['best']['source_sha256'],
            score=row['best']['display_score'],complete=row['best']['complete'],backend=row['best']['backend'],
            supplementary=row['best']['source_record'].get('extra_checks'),
            cross_safety=row['best']['source_record'].get('cross_safety'),
            results=row['best']['independent_results']) for row in index['rows']},
        targets=dict(total=30,regular_accepted=12+len(new),performance_pending=12-len(new),correctness_pending=1,special=5),
        threeway=threeway,attribution='Codex诊断/编排/源码与语义修复；CC/DeepSeek生成续搜；原KGS评测；原始失败保留')
    (batch/'delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(accepted=accepted,newly_stable=new,mechanism_facts=facts),ensure_ascii=False))


if __name__=='__main__':main()
