#!/usr/bin/env python3
"""Render original retest failures separately from independent source findings."""
import argparse
from collections import Counter
import json
from pathlib import Path

CATEGORIES = {
    'performance': '仅性能不合格', 'compiler': '编译失败（测试对象待核）',
    'accuracy': '精度/语义失败，含同时低性能',
    'resource': '共享内存超限，含同时低性能',
    'address': '非法访问/32位地址限制', 'reference': 'reference API不可用',
    'timeout': 'benchmark超时', 'test_builder': 'benchmark输入构造失败',
    'test_scope': '失败归给了不同算子', 'unspecified': '仅标错误、症状缺失',
}
VENDORS = {'tianshu': '天数', 'moer': '摩尔', 'pingtouge': 'PPU', 'muxi': '沐曦'}
STATUSES = {'located': '问题层次明确', 'related': '有相关证据，未闭环', 'pending': '待定位'}


def summarize(data, source):
    rows = data['rows']
    ids = [r['id'] for r in rows]
    old = {r['id']: r for r in source['rows']}
    if len(ids) != 44 or len(set(ids)) != 44 or set(ids) != set(old):
        raise ValueError('Must retain all 44 original rows exactly once')
    for r in rows:
        if r['primary_category'] not in CATEGORIES or r['responsibility_status'] not in STATUSES:
            raise ValueError(r['id'])
        if r['candidate_sha256'] != old[r['id']]['candidate_sha256']:
            raise ValueError('Candidate identity changed')
        if r['reported_speedup'] != old[r['id']]['reported_speedup']:
            raise ValueError('Original speedup changed')
        if r['exact_original_failure_closed']:
            raise ValueError('New exact replay evidence required before closing an original failure')
    queues = data['issue_queues_overlapping']
    for queue in queues:
        if len(queue['ids']) != len(set(queue['ids'])) or not set(queue['ids']) <= set(ids):
            raise ValueError('Invalid queue members')
    perf = {r['id'] for r in rows if r['reported_speedup'] is not None and r['reported_speedup'] < .8}
    if perf != set(next(q['ids'] for q in queues if q['id']=='PERFORMANCE')):
        raise ValueError('Performance queue must use original workbook values')
    if set().union(*(set(q['ids']) for q in queues)) != set(ids):
        raise ValueError('A mainline failure has no investigation queue')
    return {'total': len(rows), 'unique_operator_names': len({r['operator'] for r in rows}),
            'by_vendor': dict(Counter(r['vendor'] for r in rows)),
            'primary_categories_exclusive': dict(Counter(r['primary_category'] for r in rows)),
            'evidence_status_exclusive': dict(Counter(r['responsibility_status'] for r in rows)),
            'investigation_priorities_exclusive': dict(Counter(r['priority'] for r in rows)),
            'queue_counts_overlapping': {q['id']: len(q['ids']) for q in queues},
            'exact_original_failure_closed': 0}


def escape(value):
    return str(value if value is not None else '—').replace('|', '\\|').replace('\n', ' ')


def render(data, source):
    s = summarize(data, source)
    rows = {r['id']: r for r in data['rows']}
    p = lambda count: f'{count/44:.1%}'
    lines = ['# 非昇腾复测失败主线：分类、责任与优先级', '',
        '归档范围：本文及配套台账冻结于 2026-09-08 早期 Excel 的 44 条厂商/算子样本，当时海光复测数据尚未补齐。后续更新后的 62 项 pytest review 和其中 35 项重新生成、原生复测属于独立批次，不回填本历史统计；文中的“当前”均指本次历史审计时点。原始 Excel、kernel_todo_v2/ 和 runs/ 为本地实验归档，不随 Git 分发。', '',
      '**主线仍为44条厂商/算子记录、38个算子名。原表实际暴露的失败决定调查顺序；额外源码缺陷单独记账，不再计作这些失败的已确认原因。** 本文是当前主线优先级入口；[前一轮完整源码审查](../operations/troubleshooting/nonascend_retest_failure_analysis.md)保留作证据与额外发现档案，其“相关缺陷影响数”不再作为主线根因排行。', '',
      '天数9、摩尔13、PPU9、沐曦13；海光无复测数据，昇腾排除。原表数值低于0.8也算失败，不能只读FAIL字样。所有原表记录保持不变，包括已发现失败对象错配的那一行。', '',
      '## 原表失败分类（互斥，合计44）', '',
      '| 主问题 | 算子记录数 | 占比 |', '|---|---:|---:|']
    for category, count in sorted(s['primary_categories_exclusive'].items(), key=lambda x: -x[1]):
        lines.append(f'| {CATEGORIES[category]} | {count} | {p(count)} |')
    lines += ['', '精度与资源错误优先作为该行主问题，性能下降作为同一行的子问题保留。因此重叠统计是性能不合格26条、精度/语义失败6条、共享内存超限4条。原表8条编译失败报告中的scatter_add行，其失败nodeid实际属于scatter_add_；本表将它单列为对象归属错误，剩余编译队列为7条，未删除原失败记录。', '',
      '## 主线问题队列（按影响数排列，可重叠）', '',
      '表格按原表影响数量排列。P0优先处理成簇的正确性/编译/资源阻断及确定的测试对象错配；不等于这些样本已全部定责编译器、Gems或Coder。性能26条先列P1调查队列，需拆成候选变慢、reference变化、输入集合变化或测量对象错误，不能因数量大就把未知根因升级为P0。单例且缺完整复现的编译/地址/超时问题列P1；只有“错误”字样的列P2。', '',
      '| 优先级 | 原表实际问题 | 数量/占比 | 先由谁处理 | 当前证据边界 |', '|---|---|---:|---|---|']
    queues = sorted(data['issue_queues_overlapping'], key=lambda q: (-len(q['ids']), q['priority'], q['id']))
    for q in queues:
        lines.append(f"| {q['priority']} | {q['title']} | {len(q['ids'])} / {p(len(q['ids']))} | {q['owner']} | {q['limit']} |")
    lines += ['', '## 责任能确定到什么程度', '',
      '| 证据状态 | 数量 | 占比 | 含义 |', '|---|---:|---:|---|']
    meanings = {'located': '问题所在层次明确，仍不宣称原表同环境全量重跑闭环', 'related': '与原表症状有相关证据或部分解释，仍需补齐因果链', 'pending': '具体根因未定位；下一步负责人是诊断责任，不是故障定责'}
    for state in ['located', 'related', 'pending']:
        count=s['evidence_status_exclusive'].get(state,0)
        lines.append(f'| {STATUSES[state]} | {count} | {p(count)} | {meanings[state]} |')
    lines += ['', '**目前可以明确问题层次的两条：** 摩尔cudnn_convolution为MUSA Torch reference API不可用，归runtime能力与测试准备；沐曦scatter_add的原表失败节点属于scatter_add_，归复测脚本/报告的对象归属。后者实际inplace kernel的BF16编译问题仍须单独追查。不能把它写成out-of-place候选编译失败。', '',
      '其余条目都保留具体证据状态。特别是“沐曦6个BF16比较算子”：upstream lowering的最小复现已经可以提供给编译器团队，但还不能证明实习生执行的6份都是归档候选；需同时核对别名、tensor/scalar marker与实际编译源码。', '',
      '## 性能26条的归因方式', '',
      '续查见[dispatch 与其余25条性能证据](nonascend_performance_followup.md)：补回8份历史ledger，区分参数分布变化、旧reference异常和仍未解释的原表下降。', '',
      '先由评测/复测维护方恢复实际candidate与baseline身份、输入指纹、--level/计时模式及逐点双边时延。固定这些条件之后：candidate耗时上升由候选维护方定位；reference耗时变化由对应baseline/计时维护方定位；case集合或dim/stride改变由测试计划维护方记录和对齐。合法输入下的真实性能不足才进入Coder优化，不由Coder修改测试和环境。', '',
      '现有证据最具体的三条是：PPU cholesky_solve_helper的直调/dispatcher开销差异；PPU index_select_backward的dim计划差异；摩尔adaptive_avg_pool2d_backward旧best的reference异常高值。它们分别提供部分解释，没有精确闭环原表全部下降。天数expand_copy原表明确自建workload，也只能先确认比较集合不同。', '',
      '当前审计中的旁路、baseline污染、全skip和合法skip误判是获得有效重测结果前必须处理的准备问题；除非与实习生原始执行记录对齐，否则不能把它们计作原表降速的已确认根因。特别是0.0039×/0.0047×，既不能凭数值认定填错列，也不能凭跌幅认定编译器回归。', '',
      '## 精度6条的分工', '',
      '| 厂商/算子 | 当前主线问题 | 责任判断 |', '|---|---|---|',
      '| 天数 special_bessel_y0 | FP32扩展输入失败 | 候选域外反例与报告相关；先取得4个失败输入，allocator问题另案 |',
      '| 天数 shifted Chebyshev T | 10个精度/语义失败点 | n>12反例是线索，不能认定10点全由此导致 |',
      '| 天数 baddbmm_ | 非默认scalar精度失败，同时低性能/资源问题 | 候选错误分支与Gems输入污染分工处理；核对原表reference输入是否隔离 |',
      '| 摩尔 matmul_bias_activation | FP32 M495/N5333/K71精度失败，同时低性能 | 同SHA旧best该shape通过；先固定数值输入与实际执行对象，不以新增shape解释 |',
      '| 摩尔 AA backward | ref稀疏/res稠密 | 原表点名kernel不在交付候选，先查对象和独立gold，不预设ref正确 |',
      '| PPU conv_depthwise2d | 只有正确性不过说明 | 先取原始数值错误，后续私有编译器API导入问题不能替代它 |', '',
      '## 全部44条主线台账', '']
    for vendor in VENDORS:
        lines += [f'### {VENDORS[vendor]}', '', '| 行ID / 算子 | 原表状态 / 加速比 | 级别 / 证据 | 下一步责任方 | 主线判断 |', '|---|---|---|---|---|']
        for r in data['rows']:
            if r['vendor']!=vendor:continue
            speed='—' if r['reported_speedup'] is None else f"{r['reported_speedup']:g}×"
            lines.append(f"| {r['id']} `{r['operator']}` | {escape(r['reported_status'])} / {speed} | {r['priority']} / {STATUSES[r['responsibility_status']]} | {r['responsible_next_step']} | {r['mainline_assessment']} |")
        lines.append('')
    lines += ['## 额外发现：记录但不进入主线根因计数', '',
      '候选修改cache/编译环境/shim/注册、沐曦less_equal_广播错误、PPU linear和天数addmm_等与原表症状尚未关联的边界缺陷，都保留在原源码台账中。对有潜在关联的域外公式、scalar错误或matmuladd越界，仅作为主线线索，未升级为已闭环根因。', '',
      '| 行ID / 算子 | 源码发现 | 本轮用途 |', '|---|---|---|']
    for extra in data['supplemental_source_findings']:
        row=rows[extra['id']]
        use='相关排查线索，未闭环' if extra['mainline_use']=='related_lead_not_closed' else '额外待办，不计原表失败根因'
        lines.append(f"| {extra['id']} `{row['operator']}` | {extra['finding']} | {use} |")
    lines += ['', '## 工程职责边界', '',
      '测试准备/pytest review负责reference可执行性、实际注入对象、baseline隔离、合法skip和case计划；Coder只修候选算法、访存、数值和合法配置下的性能。编译器、reference或注入系统的错误，由对应维护者处理，Coder提供证据不越权修补。普通候选Triton错误或tile超硬件资源上限仍由候选维护方处理，不能见编译报错就BLOCK给编译器团队。', '',
      '本轮仅重新整理已有证据与优先级，没有新增GPU运行，没有修复候选、测试、框架或运行时。原表同条件闭环仍为0；这不是说没有发现确定问题，而是明确区分“问题层次可定位”“部分机制解释”和“同条件复现并验证修复”。不使用此前31条“相关源码/审计问题”的数量作为主线完成率。', '',
      '[主线机器台账](../operations/data/nonascend_retest_mainline_20260908.json)；[自动统计](../operations/data/nonascend_retest_mainline_summary_20260908.json)；[原始证据与额外发现台账](../operations/data/nonascend_retest_failure_20260908.json)。原始Excel与设备日志均为本地实验归档。', '',
      '```bash',
      'python3 scripts/analysis/render_nonascend_mainline.py --ledger docs/operations/data/nonascend_retest_mainline_20260908.json --source docs/operations/data/nonascend_retest_failure_20260908.json --report docs/validation/nonascend_retest_mainline.md --summary docs/operations/data/nonascend_retest_mainline_summary_20260908.json',
      'python3 -m unittest discover -s tests/analysis -p "test_nonascend*.py"',
      '```', '']
    lines[1:1] = ['', '> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['ledger','source','report','summary']:
        parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.ledger.read_text());source=json.loads(args.source.read_text())
    summary=summarize(data,source)
    args.report.write_text(render(data,source))
    args.summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
