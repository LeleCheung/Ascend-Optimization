#!/usr/bin/env python3
"""从最终交付与同批父版逐case延迟生成中文简报，不按汇总ratio相除估算收益。"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    args = parser.parse_args()
    b = args.batch_dir
    tools = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('buffer_export', tools/'export-paired-deliverables.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    delivery = json.loads((b/'deliverables-final/manifest.json').read_text())
    audit = json.loads((b/'completion-audit.json').read_text())
    if audit['status'] != 'passed':raise ValueError('最终证据未通过验收')
    records = []
    for row in delivery['records']:
        op = row['operator']
        selected = row['selected']
        parent = exporter.read_arm(b, op, 'codex-parent', require_dense=True)
        if parent is None:raise ValueError('缺少同批有效父版')
        current_result = json.loads(Path(selected['evidence']['result']).read_text())
        parent_result = json.loads(Path(parent['evidence']['result']).read_text())
        timing = {r['uuid']: r for r in current_result['per_workload'] if r['phase'] == 'timing'}
        old_timing = {r['uuid']: r for r in parent_result['per_workload'] if r['phase'] == 'timing'}
        if set(timing) != set(old_timing):raise ValueError('父版与最终case不相同')
        cases = [dict(case_id=k, parent_us=old_timing[k]['latency_ms']*1000,
            final_us=timing[k]['latency_ms']*1000, reference_us=timing[k]['reference_latency_ms']*1000,
            relative_to_parent=old_timing[k]['latency_ms']/timing[k]['latency_ms'],
            reference_parent_us=old_timing[k]['reference_latency_ms']*1000) for k in sorted(timing)]
        factor = exporter.geo([r['relative_to_parent'] for r in cases])
        if abs(factor-parent['geometric_latency_ms']/selected['geometric_latency_ms']) > 1e-10:
            raise ValueError('逐case与整体实际耗时比不符')
        comparisons = []
        for index in [0,3]:
            f = b/'research/final-final'/('%s-paired-%d'%(op,index))/'comparison.json'
            d = json.loads(f.read_text())
            if d['spec']['source_sha256'] != selected['source_sha256']:raise ValueError('成对采集源码不同')
            comparisons.append(dict(case_id=d['spec']['case_id'],
                reference_us=float(d['reference']['basic']['Task Duration(us)']),
                candidate_us=float(d['candidate']['basic']['Task Duration(us)']),
                counters=[r for r in d['differences'] if r['metric'] in
                          ['aiv_mte2_instructions','aiv_mte3_instructions','GM_to_UB_datas(KB)']],
                derived=d['derived_differences'],path=str(f)))
        records.append(dict(operator=op, selected=selected, parent=parent, cases=cases,
            actual_speedup_over_parent=factor, actual_latency_reduction_percent=(1-1/factor)*100,
            paired= comparisons))
    output = dict(records=records, audit_checks=audit['checks'],
        all_meet_08=delivery['all_meet_08'], timing_scope='KGS原36项完整调用walltime',
        selection='同case候选实际延迟几何平均最小；不以参考变慢选择源码')
    (b/'final-comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['# log10_／selu_ 本轮精简汇报', '',
        '| 算子 | Excel原记录 | 固定FlagGems实测 | 本轮父版 | HG-KDA最终 | 相对父版实际加速 |',
        '|---|---|---|---:|---:|---:|']
    for r in records:
        fixed = '正确性失败，33/36' if r['operator']=='log10_' else '0.119×'
        lines.append('| `%s` | 精度失败 | %s | %.3f× | **%.3f×** | %.3f× |' %
            (r['operator'],fixed,r['parent']['geo_mean'],r['selected']['geo_mean'],r['actual_speedup_over_parent']))
    lines += ['', '表内性能均相对PyTorch；相对父版实际加速由同case候选耗时计算。FlagGems为固定提交的历史实测；父版和最终版为本批同合同、各一次正式评测。', '',
        '两项最终源码均通过36项正式测试、42项布局/特殊值、12项大连续输入和9项数值范围检查；补充检查每项两次并切换stream。最终原生KGS profiling和四组成对采集完成，%d项证据核验通过。'%audit['checks'], '',
        '最终来源：'+ '；'.join('`%s`：`%s`，源码SHA `%s`'%(r['operator'],r['selected']['arm'],r['selected']['source_sha256']) for r in records)+'。', '',
        'Codex负责既有硬件证据诊断、编译选项试验、候选整合与独立验收；CC/DeepSeek负责追加源码生成。目录名中的`codex-`表示独立验证路径，具体源码贡献以provenance为准。', '',
        '详细逐case耗时和成对硬件结果见[final-comparison.json](final-comparison.json)，复现命令见[README](README.md)。']
    for r in records:
        if r['selected']['source_sha256']==r['parent']['source_sha256']:
            lines += ['', '`%s`沿用上一轮源码，本轮新候选未改善整组实际耗时；本批复测分数单列，上轮历史成绩保持原样。'%r['operator']]
    (b/'精简汇报.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='records'},ensure_ascii=False))


if __name__ == '__main__':
    main()
