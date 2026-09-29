#!/usr/bin/env python3
"""Validate the reviewed ledger and render counts and every affected operator."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path

VENDORS = {'tianshu': '天数', 'moer': '摩尔', 'pingtouge': 'PPU', 'muxi': '沐曦'}
SYMPTOMS = {
    'performance': '仅性能低于0.8，未报告其他执行错误',
    'compiler': '编译失败（含原表误写成精度失败的编译错误）',
    'accuracy': '精度/语义失败，无低于0.8的有效数值',
    'accuracy_performance': '精度失败且性能低于0.8（可另伴资源错误）',
    'resource': '共享内存超限（可另伴已完成点低性能）',
    'address': '非法内存访问或32位地址限制',
    'reference': 'reference API不可运行',
    'timeout': 'benchmark超时',
    'test_builder': 'benchmark输入生成失败',
    'unspecified': '只有错误标记、缺具体症状',
}
CANDIDATE_GROUPS = {'STATE', 'BOUNDS', 'DOMAIN', 'INPUT_CACHE', 'AA_REF'}
EVAL_GROUPS = {'IDENTITY', 'BASELINE', 'REF_SPIKE', 'NO_TESTS', 'LEGAL_SKIP', 'DISPATCH', 'WORKLOAD', 'INPUT_ALIAS', 'BUILDER'}


def validate(d):
    rows = d['rows']
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate vendor/operator row')
    if len(ids) != d['count_summary']['total']:
        raise ValueError('Incomplete cohort')
    group_map = {g['id']: g for g in d['groups']}
    for g in d['groups']:
        members = g['confirmed_ids'] + g['reported_only_ids']
        if len(members) != len(set(members)) or not set(members) <= set(ids):
            raise ValueError(f"Invalid group membership: {g['id']}")
    for r in rows:
        if r['symptom'] not in SYMPTOMS:
            raise ValueError(f"Missing symptom: {r['id']}")
        expected = {g['id'] for g in d['groups'] if r['id'] in g['confirmed_ids'] + g['reported_only_ids']}
        confirmed = {g['id'] for g in d['groups'] if r['id'] in g['confirmed_ids']}
        if set(r['groups']) != expected or set(r['confirmed_groups']) != confirmed:
            raise ValueError(f"Asymmetric membership: {r['id']}")
        if r['original_failure_closed']:
            raise ValueError('Closure requires a new reviewed exact-replay evidence schema')
        if r['priority'] != min((group_map[x]['priority'] for x in expected), default='P1'):
            raise ValueError(f"Inconsistent priority: {r['id']}")
    return rows


def stats(d):
    rows = validate(d)
    n = len(rows)
    def members(groups):
        return {r['id'] for r in rows if groups & set(r['confirmed_groups'])}
    candidates = members(CANDIDATE_GROUPS)
    evaluation = members(EVAL_GROUPS)
    compiler = members({'BF16'})
    evidence = candidates | evaluation | compiler
    return {
        'total': n, 'unique_operator_names': len({r['operator'] for r in rows}),
        'by_vendor': dict(Counter(r['vendor'] for r in rows)),
        'symptoms_exclusive': dict(Counter(r['symptom'] for r in rows)),
        'priorities_exclusive': dict(Counter(r['priority'] for r in rows)),
        'with_confirmed_related_issue': len(evidence),
        'without_confirmed_related_issue': n-len(evidence),
        'exact_original_failure_closed': sum(r['original_failure_closed'] for r in rows),
        'confirmed_issue_sets_overlapping': {'candidate': sorted(candidates), 'evaluation_or_comparability': sorted(evaluation), 'vendor_compiler_related': sorted(compiler)},
        'group_counts': {g['id']: {'confirmed': len(g['confirmed_ids']), 'reported_only': len(g['reported_only_ids']), 'total': len(g['confirmed_ids'])+len(g['reported_only_ids'])} for g in d['groups']},
    }


def verify_local_evidence(d):
    verified = []
    for r in validate(d):
        source = Path(r['candidate_path'])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != r['candidate_sha256']:
            raise ValueError(f"Candidate identity mismatch: {r['id']}")
        result_path = Path(r['prior_native']['result_path'])
        result = json.loads(result_path.read_text())
        if result['candidate_sha256'] != digest:
            raise ValueError(f"Native result candidate mismatch: {r['id']}")
        if result['native_commit'] != d['provenance']['prior_native_gems']:
            raise ValueError(f"Native Gems revision mismatch: {r['id']}")
        if result.get('benchmark_level') != 'core':
            raise ValueError(f"Unexpected benchmark level: {r['id']}")
        verified.append({'id': r['id'], 'candidate_sha256': digest,
                         'native_result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest()})
    return verified


def cell(value):
    return str(value or '—').replace('|', '\\|').replace('\n', ' ')


def render(d):
    s = stats(d)
    n = s['total']
    pct = lambda count: f'{count/n:.1%}'
    lines = ['# 非昇腾：历史通过、复测失败的完整归因台账（2026-09-08）', '',
        '归档范围：本文及配套台账冻结于 2026-09-08 早期 Excel 的 44 条厂商/算子样本，当时海光复测数据尚未补齐。后续更新后的 62 项 pytest review 和其中 35 项重新生成、原生复测属于独立批次，不回填本历史统计；文中的“当前”均指本次历史审计时点。原始 Excel、kernel_todo_v2/ 和 runs/ 为本地实验归档，不随 Git 分发。', '',
        '当前主线分类与优先级见[复测实际失败主线](../../validation/nonascend_retest_mainline.md)。本文保留完整源码审查；以下相关缺陷数量不能用作原表失败根因的影响数。', '',
        f"**共 {n} 条厂商／算子样本，去重 {s['unique_operator_names']} 个算子名。所有 {n} 条都已逐份核查候选源码、原表描述与已有原生证据；{s['with_confirmed_related_issue']} 条存在已证实的相关缺陷或评测差异，另外 {s['without_confirmed_related_issue']} 条仍只有故障报告或待验证假设。** 这里的“相关问题已证实”包含新发现的独立源码缺陷，不等于原表失败已闭环。实习生原始命令、失败输入、实际执行代码和完整环境未齐备，本轮没有将任何条目标为原表同条件复现完成。", '',
        '本报告替代之前只围绕已有 P0 标签的覆盖估计。没有把当前 `--level core` 通过当作原表失败消失，也没有把表里的“建议修法”直接当作已经验证的根因。全部原始单元格、候选 SHA、责任范围、证据限制与下一步见 [机器台账](../data/nonascend_retest_failure_20260908.json)。', '',
        '## 计数口径', '',
        '选取 V2 归档状态为“成功”，且原 Excel 的 D/H 栏报告失败，或 E 栏数值加速比低于 0.8 的记录。正确性 PASS 但性能不合格仍算复测失败；没有有效计时也不能凭正确性通过验收。PPU 不能只读 D 栏。原表旧状态的 PASS/成功/达标与归档身份交叉核对。', '',
        '| 厂商 | 样本数 | 占44条比例 |', '|---|---:|---:|']
    for vendor, count in s['by_vendor'].items():
        lines.append(f'| {VENDORS[vendor]} | {count} | {pct(count)} |')
    lines += ['', '海光工作表没有复测数据，不能称为“零失败”；昇腾完全排除。29 条有明确失败状态，26 条数值低于0.8，其中11条重叠，所以并集为44条。之前只算29条漏掉15条性能不合格记录。摩尔页末的8条环境说明不是算子。35份候选可关联V2固定Gems版本，另外9份继承历史V1候选、旧revision尚未完整核验，不能一律写成同一旧版本生成。', '',
        '## 全部失败的现象分布（互斥，合计44）', '',
        '优先把“精度失败且低性能”归入混合项；其余共享内存超限单列，BF16 PassManager失败按编译分类。这里描述原表症状，不是责任方占比。', '',
        '| 原表症状 | 数量 | 占比 |', '|---|---:|---:|']
    for name, count in sorted(s['symptoms_exclusive'].items(), key=lambda x: -x[1]):
        lines.append(f'| {SYMPTOMS[name]} | {count} | {pct(count)} |')
    lines += ['', '另按重叠口径：26条存在数值性能不合格；6条报告数值/语义精度失败；4条报告共享内存超限。原表8条编译失败中，7条为沐曦BF16报告（6个比较算子及1个实际inplace scatter_add_），1条为摩尔LLVM崩溃。不能把这些全叫“候选精度不过”。', '',
        '## 头部问题及责任（按证实影响数排序）', '',
        '优先级用于安排修复/反馈，不表示因果确定性。P0优先处理能污染结论、破坏正确性或跨多个算子复用的缺陷；同级按受影响算子行数排序。仅报错而未定位的同类大簇单列“待确认”，不能借症状数量扩大已确认责任。一个样本可有多个问题，下表比例不能相加。', '',
        '| 优先级／问题 | 已证实相关样本 | 占比 | 另待确认 | 责任仓库／层次 | 最小处理方案 |',
        '|---|---:|---:|---:|---|---|']
    groups = sorted(d['groups'], key=lambda g: (-len(g['confirmed_ids']), g['priority'], g['id']))
    for g in groups:
        if not g['confirmed_ids']:
            continue
        count = len(g['confirmed_ids'])
        lines.append(f"| {g['priority']} {g['title']} | {count} | {pct(count)} | {len(g['reported_only_ids'])} | {g['owner']} | {g['solution']} |")
    lines += ['', '源码候选问题涉及17/44条（38.6%），评测身份、输入、计时或可比性问题涉及17/44条（38.6%），与已有编译器最小复现直接关联2/44条（4.5%）。这些集合重叠，去重后31条（70.5%）；余13条（29.5%）没有形成独立确定机制。**这些数字不能改写成“70.5%的原表失败已定位”，也不能把全部评测差异都定责KGS。** Gems upstream、KG复测工具、KGS adapter、候选与厂商工具链的具体职责在逐项记录中分开。', '',
        '| 仅报告或尚未确认的大簇 | 样本数 | 占比 | 当前分级 | 责任边界 |', '|---|---:|---:|---|---|',
        '| 沐曦BF16比较编译簇 | 6（2条有关联实证，4条待对齐） | 13.6% | P0待确认 | upstream lowering最小复现已可反馈；不宣称6个交付候选均回归 |',
        '| 天数共享内存超限 | 4 | 9.1% | P0待确认 | 先确定candidate/reference与tile资源；超硬件上限本身不等于编译器bug |',
        '| 纯性能降幅仍无确定缺陷解释的重点队列 | 6 | 13.6% | P1 | 摩尔SVD、BN、LSTM backward、masked accumulate、nearest exact backward；PPU baddbmm_ |',
        '| 摩尔LLVM后端崩溃 | 1 | 2.3% | P1待复现 | 原表llc -6；当前精确候选通过，待原IR与版本 |',
        '| 沐曦32位地址限制 | 1 | 2.3% | P1待定位 | 候选偏移提升与后端限制尚未分开 |', '',
        '“6条纯性能重点队列”只表示没有找到独立确定缺陷的性能样本；其他已有副作用、旁路或源码错误的算子，其原表性能跌幅也可能没有解释完。不能将其余性能问题当作已经解决。', '',
        '## 本次新增且不能忽略的证据', '',
        '五份候选缺访存边界：天数 addmm_，摩尔 matmul_bias_activation，PPU linear，沐曦 matmuladd、matmul_bias_activation。host地址计算给出每条启用的越界访问；摩尔使用原表记载的 M495/N5333/K71 就进入缺B上界mask的转换路径。源码缺陷已成立，但是否在某次GPU分配下触发非法访问仍需设备验证。', '',
        '此前漏标的全局状态修改包括PPU unbind_copy写入llc shim并改编译工具路径、PPU float_power_持久注册context、沐曦linear移除编译环境变量。摩尔cudnn_convolution还有按张量对象缓存repack/im2col的问题：同一张量内容改变后返回陈旧结果，且重复benchmark省略预处理。合法的常量元数据缓存没有归入此项。', '',
        '摩尔AA backward交付候选的整倍数下采样梯度是稀疏scatter。16→8且output梯度在(3,3)处为1时，候选只向(6,6)给1；独立bilinear antialias导数应分布到16个输入像素，(6,6)为9/64。与此同时，原表所称的fused kernel只在Gems upstream存在，并不在交付候选里。不能按原表“ref稀疏，所以res稠密错了”的推断继续修改候选；需要核对CPU gold、MUSA Torch、Gems及候选四方。', '',
        '天数baddbmm_的输入别名问题由Gems upstream测试顺序引入；已有git blame证明KG后续resolver补丁没有改动reference/clone顺序。候选为拟合这一行为直接返回输入，则属于我方独立错误。', '',
        '## 性能下降的判断边界', '',
        '最隐蔽的已确认问题是reference异常高值进入历史best、候选替换同时污染baseline，以及直调/ATen入口不同。它们均可能在“正确性通过、候选有调用、计时为正”时让加速比失真。摩尔adaptive_avg_pool2d_backward的旧reference单点异常1208.7倍，但这仍不足以解释原表0.0619×；LSTM backward的0.0039×和masked accumulate的0.0047×也不能在缺原始行时被认定为填错列或编译器回归。', '',
        '已核对的原生Gems commit中，benchmark未指定`--level`时默认comprehensive；我们的复测显式core。天数230/242个性能点与当前15点、沐曦30shape与当前15点不能直接等价比较。它可能来自默认level、layout或额外workload，但实习生完整命令未归档，不能写成“已经证明Gems更新新增了这些workload”。35份可关联V2候选的直接参数装饰器对比也不足以排除间接shape、随机dim和输入内容变化。', '',
        '后续复测必须按每个case保留baseline_ms、candidate_ms、完整参数/stride指纹、实际callable/hash和计时模式。先比较共同case的分子/分母变化，再单列新增case；固定core不意味着删除或忽略原表扩展输入失败。', '',
        '## 当前preflight／pytest review能保证什么', '',
        '用KGS `b4aad01` 的flaggems源码准入重新扫描全部44份，拒绝3份：摩尔cudnn_convolution、var，PPU conv_depthwise2d。它们属于8份全局状态问题中的3份，另外5份确实没有被当前源码规则识别。41份未拒绝不能统一称为“41个漏报”：纯性能、数学语义、硬件资源和实际注入身份不都是静态源码门禁的职责。', '',
        'pytest review应负责测试准备、调用对象、baseline隔离、有效测试及输入生成；preflight负责候选源码准入；完整正确性与性能验收承担语义和计时检查。本轮没有新增算子专用规则，没有扩大BLOCK职责，也没有把review skill的静态无提示当作已经完成动态review。', '',
        '## 全部44条逐项记录', '']
    for vendor in VENDORS:
        lines += [f'### {VENDORS[vendor]}', '', '| 行ID／算子／优先级 | 原表结果 | 责任与已核实问题 | 未闭环范围及下一步 | 证据 |', '|---|---|---|---|---|']
        for r in d['rows']:
            if r['vendor'] != vendor:
                continue
            c = r['cells']
            source = f"[候选]({r['candidate_path']})"
            native = r['prior_native']['result_path']
            evidence = source + (f'；[既有原生记录]({native})' if native else '')
            before = cell(c.get('C'))
            after = cell(c.get('E'))
            result = f"{before} → {after}；{cell(c.get('D') or c.get('H'))}"
            finding = f"**{r['responsibility']}**。{r['finding']}"
            limit = f"{r['limit']} **下一步：**{r['next_step']}"
            lines.append(f"| {r['id']} `{r['operator']}` {r['priority']} | {result} | {cell(finding)} | {cell(limit)} | {evidence} |")
        lines.append('')
    pending = [f"{r['id']} `{r['operator']}`" for r in d['rows'] if not r['confirmed_groups']]
    lines += ['## 证据与复现', '',
        '未形成独立确定机制的13条完整名单：'+ '；'.join(pending)+'。其余31条的原表同条件闭环限制仍必须阅读逐项台账，不能视为已修复。', '',
        f"本轮新增GPU运行：{d['provenance']['new_gpu_runs']}。使用已保存且核对候选SHA的原生结果，新增host源码/数学反例与源码门禁扫描；没有安装或替换Torch、Triton或厂商运行时，没有启动远端Server/Coder。它是完整样本归因审核，不是44条新镜像同条件重跑。", '',
        '分析基线：KG v6.2.1@0e595c72；源码门禁KGS feature b4aad01；已有原生复测Gems 34bd6d68928c8b0039c42987840929da52aa6a62。没有创建新KG/KGS/Protocol发布组合。原表与归档映射SHA见机器台账。', '',
        f"host反例：[结果]({d['provenance']['host_probes']})；[脚本](../../../scripts/analysis/probe_nonascend_candidates.py)。源码门禁：[44条输出]({d['provenance']['preflight_result']})。候选源码只读取，数学探针不冒充设备编译或原表失败输入重放。", '',
        '原始Excel和kernel_todo_v2是本机实验输入，不随Git分发。恢复同SHA输入后可重建计数；已审核的归因台账随本分支保存，统计和Markdown可不依赖GPU重建：', '',
        '```bash',
        'python3 scripts/analysis/analyze_nonascend_retest.py --workbook <复测.xlsx> --attribution <旧身份映射/attribution.json> --output <cohort.json>',
        'python3 scripts/analysis/probe_nonascend_candidates.py --cohort <cohort.json> --output <source-counterexamples.json>',
        'python3 scripts/analysis/render_nonascend_retest.py --ledger docs/operations/data/nonascend_retest_failure_20260908.json --report docs/operations/troubleshooting/nonascend_retest_failure_analysis.md --summary docs/operations/data/nonascend_retest_failure_summary_20260908.json',
        'python3 -m unittest discover -s tests/analysis -p "test_nonascend*.py"',
        '```', '',
        '优先开发顺序：先补共享的pytest身份/基线review与全局状态准入缺口；随后处理5份边界缺陷和明确语义反例；并行准备厂商BF16/LLVM/资源问题的证据包，尚未同条件复现的只作为待确认反馈。按用户既定约束，确需重新生成时使用Codex runtime、传入旧Triton作为reference Triton；当前任务未启动新一轮生成。', '']
    lines[1:1] = ['', '> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--verify-local-evidence', type=Path, help='Optional output receipt; verifies frozen local source/native identities')
    args = parser.parse_args()
    data = json.loads(args.ledger.read_text())
    summary = stats(data)
    if args.verify_local_evidence:
        receipt = verify_local_evidence(data)
        args.verify_local_evidence.write_text(json.dumps(receipt, indent=2)+'\n')
    args.report.write_text(render(data))
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'group_counts' and k != 'confirmed_issue_sets_overlapping'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
