#!/usr/bin/env python3
"""复测既有对照并输出持续优化批次的源码、逐用例与简洁中文报告。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from optimization_campaign import PROJECT, EVAL_TOOLS, checked, digest, evaluate, save


def local_result(root, original):
    name = original.replace('\\', '/')
    return root / 'reports' / name.split('/reports/', 1)[1]


def replay_report(operator, campaign_batch):
    """报告新批次本身的复验，不与提交前的旧合同混合。"""
    root = PROJECT / 'operators' / operator
    campaign = root / 'reports' / ('profiling-agent-' + campaign_batch)
    state = json.loads((campaign / 'campaign-state.json').read_text(encoding='utf-8'))
    if state.get('status') != 'completed' or state.get('workflow_kind') != 'historical-replay':
        raise ValueError('历史复现尚未完成')
    seed, seed_score = checked(campaign / 'seed-evaluation/seed-1.result.json')
    if seed['source_sha256'] != state['seed_sha256']:
        raise ValueError('复现起点源码不匹配')
    scores = []
    for index in (1, 2):
        result, score = checked(campaign / 'final-evaluation' / f'final-{index}.result.json', seed)
        if (result['source_sha256'] != state['final_source_sha256']
                or not math.isclose(score, state['final_scores'][index - 1], rel_tol=1e-10)):
            raise ValueError('最终复验源码或分数不匹配')
        scores.append(score)
    if digest((campaign / 'final.py').read_text(encoding='utf-8')) != state['final_source_sha256']:
        raise ValueError('最终源码文件不匹配')
    reused = state['final_source_sha256'] == state['seed_sha256']
    lines = [f'# {operator} 历史方法复现', '',
             f'完整{len(seed["rows"])}项通过。两次相对PyTorch为**{scores[0]:.3f}×、{scores[1]:.3f}×**，'
             f'采用较慢值**{min(scores):.3f}×**，计时范围为设备kernel。', '',
             f'起点{seed_score:.3f}×；最终源码{"与历史起点一致" if reused else "包含增量修改"}。'
             '流程完成真机采集、中文诊断、方法/提示词保存和完整两次复验。', '',
             f'源码：[final.py](final.py)，SHA256：`{state["final_source_sha256"]}`。', '',
             f'批次：`{campaign_batch}`；本批仓库HEAD：`{state["repo_commit"]}`；'
             f'知识SHA256：`{state["configuration"]["knowledge_sha256"]}`。', '',
             '本报告只使用本批完整合同。提交或环境变化后可用新批次复现，历史四版本报告继续保留在原合同下。', '']
    path = campaign / '复现汇报.md'
    path.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print('REPLAY_REPORT_COMPLETED', str(path), scores, flush=True)
    return path


def finalize(operator, batch, prior, *, offline=False, campaign_batch=None):
    root = PROJECT / 'operators' / operator
    campaign_batch = campaign_batch or batch
    campaign = root / 'reports' / ('profiling-agent-' + campaign_batch)
    state = json.loads((campaign / 'campaign-state.json').read_text(encoding='utf-8'))
    if state['status'] != 'completed':
        raise ValueError('优化尚未完成：' + operator)
    replay = state.get('workflow_kind') == 'historical-replay'
    closure = root / 'reports' / ('master-closure-' + batch)
    closure.mkdir(parents=True, exist_ok=True)
    arguments = [sys.executable, str(EVAL_TOOLS / 'compare-operator-versions.py')]
    if prior:
        baseline = json.loads((root / 'reports' / prior / '版本对比.json').read_text(encoding='utf-8'))
        for row in baseline[:3]:
            original = local_result(root, row['result_path'])
            label = original.name.removesuffix('.result.json').rsplit('-', 1)[0]
            source = original.with_name(label + '.py')
            directory = closure / 'baselines' / row['version']
            result = evaluate(operator, source, directory, row['version'], 'http://127.0.0.1:19655',
                              offline=offline)
            arguments += ['--version', row['version'], str(result)]
    else:
        screening = json.loads((PROJECT / 'operators' / ('next-operator-screen-' + batch + '.json')).read_text(encoding='utf-8'))
        master = next(row for row in screening if row.get('selected') and row['operator'] == operator)
        arguments += ['--version', 'master', str(local_result(root, master['result']))]
        for arm in ('no-profile', 'profile'):
            directory = root / 'reports' / ('kg-' + arm + '-' + batch) / 'independent'
            scores = [(i, checked(directory / f'kg-{arm}-{i}.result.json')[1]) for i in (1, 2)]
            repeat = min(scores, key=lambda item: item[1])[0]
            arguments += ['--version', 'kg-' + arm, str(directory / f'kg-{arm}-{repeat}.result.json')]
    index = min(enumerate(state['final_scores'], 1), key=lambda item: item[1])[0]
    final = campaign / 'final-evaluation' / f'final-{index}.result.json'
    arguments += ['--version', 'workflow-replay' if replay else 'agent', str(final),
                  '--output', str(closure / '版本对比.md')]
    subprocess.run(arguments, check=True)
    final_code = root / 'candidates' / ('profiling-agent-final-' + campaign_batch + '.py')
    final_code.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(campaign / 'final.py', final_code)
    rows = json.loads((closure / '版本对比.json').read_text(encoding='utf-8'))
    seed, seed_score = checked(local_result(root, state['seed_result']))
    best, _ = checked(final, seed)
    changes = sorted([(seed['timing'][case]['latency_ms'] / row['latency_ms'], case, row)
                      for case, row in best['timing'].items()], reverse=True)
    lines = [f'# {operator} 简洁汇报', '',
             f'完整 {len(best["rows"])} 项测试通过。相对 PyTorch 的设备 kernel 加速比为 **{min(state["final_scores"]):.3f}×**；'
             f'两次复验为 {state["final_scores"][0]:.3f}×、{state["final_scores"][1]:.3f}×，相对固定 master 为 **{rows[-1]["relative_master"]:.2f}×**。', '',
             f'自动流程从本轮起点 {seed_score:.3f}× 开始，采集最慢用例的真机 msprof，结合语义、源码、910B 知识与历史失败让 CC+DeepSeek 生成候选；以完整合同决定是否保留。', '',
             '| 版本 | 相对 PyTorch | 相对固定 master |', '| --- | ---: | ---: |']
    if replay:
        lines[4] = (f'本批将历史有profiler版本及诊断方法固化为自动workflow，从已验证源码 '
                    f'{seed_score:.3f}× 出发，重新采集真机msprof、保存诊断和方法并完整复验。'
                    '目标是复现既有成果，未采纳新候选时不宣称产生了独立的新优化。')
        labels = {'master': '固定master', 'kg-no-profile': '历史无profiler流程',
                  'kg-profile': '历史有profiler流程', 'workflow-replay': '历史方法workflow复现'}
    else:
        labels = {}
    for row in rows:
        lines.append(f'| {labels.get(row["version"], row["version"])} | {row["relative_pytorch"]:.3f}× | {row["relative_master"]:.2f}× |')
    lines += ['', ('三个逐case复测值（相同源码时差异属于运行波动）：' if replay else
                   '相对本轮起点改善最大的三个 timing 用例：'), '']
    for ratio, case, row in changes[:3]:
        before = seed['timing'][case]
        lines.append(f'- `{row["axes"]}`：{before["latency_ms"] * 1000:.2f}μs → '
                     f'{row["latency_ms"] * 1000:.2f}μs（{ratio:.2f}×）。')
    valid = [row for row in state['history'] if row.get('status') == 'PASSED']
    if campaign_batch != batch:
        lines += ['', f'本次workflow使用批次`{campaign_batch}`；对照来自`{batch}`。'
                  '续跑包含额外生成与评测成本，具体起点和人工修复见算子目录的续跑说明；该表展示完整复验成绩，不代表相同总预算的消融。']
    if replay:
        lines += ['', '历史方法与来源见`../../../../tools/profiling/历史复现知识-20261014.md`。'
                  'Codex参与诊断整理与方法固化；历史KG两组名称表示运行开关，不表示整个工程未经Codex参与。'
                  f'本次最终源码{"与历史起点相同" if state["final_source_sha256"] == state["seed_sha256"] else "采纳了增量修改"}。']
    changes_by_operator = {
        'matmul_bias_activation': '关键修改：把双 N tile 复用 A 的路径扩展到 FP32 规则方阵，FP32 K 分块取128，半精度保留256；合并被新分流覆盖的旧路径。',
        'amin': '关键修改：尝试 BF16 原生精度归约，保留编译失败时的FP32回退；整除维度去掉边界掩码，并启用多缓冲与逐配置单缓冲回退。',
    }
    if operator in changes_by_operator:
        lines += ['', changes_by_operator[operator]]
    lines += ['', '| dtype | 相对 PyTorch | 最慢 case 加速比 |', '| --- | ---: | ---: |']
    by_dtype = {}
    for row in best['timing'].values():
        by_dtype.setdefault(row['axes']['dtype'], []).append(row['reference_latency_ms'] / row['latency_ms'])
    for dtype, values in sorted(by_dtype.items()):
        lines.append(f'| {dtype} | {statistics.geometric_mean(values):.3f}× | {min(values):.3f}× |')
    accepted = [row['round'] for row in valid if row.get('accepted')]
    round_summary = (f'记录 {len(state["history"])} 轮诊断/尝试，采纳轮次 {accepted}；'
                     '纯复现轮不生成新源码，额外生成尝试及失败按原记录保留。' if replay else
                     f'记录 {len(state["history"])} 轮，其中 {len(valid)} 轮完整通过，采纳轮次 {accepted}；'
                     '生成、编译与正确性失败均保留在原批次。')
    lines += ['', round_summary + '最终参数和关键改进以候选源码为准。', '',
              f'最终源码：`../../candidates/{final_code.name}`。逐用例对照见 `版本对比.csv`；'
              f'诊断、模型响应和失败历史见 `../{campaign.name}/`。', '',
              '数值 Roofline 尚未建立；本次使用真机流水线指标与算子语义指导修改。模型提议里的带宽或容量推算属于假设，最终性能以完整测试为准。', '']
    (closure / '精简汇报.md').write_text('\n'.join(lines), encoding='utf-8')
    save(closure / '最终候选.json', {'operator': operator, 'batch': batch, 'campaign_batch': campaign_batch,
         'workflow_kind': state.get('workflow_kind', 'optimization'),
         'source': final_code.relative_to(PROJECT).as_posix(), 'source_sha256': state['final_source_sha256'],
         'results': [str(campaign / 'final-evaluation' / f'final-{i}.result.json') for i in (1, 2)],
         'final_scores': state['final_scores'], 'selected_repeat': index})
    print('FINALIZED_AGENT_CAMPAIGN', operator, state['final_scores'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator', required=True)
    parser.add_argument('--batch', default='20261012-agent')
    parser.add_argument('--campaign-batch', help='续跑Agent的批次，原生KG对照仍使用--batch')
    parser.add_argument('--prior-closure', help='既有对照的报告目录名；省略时读取新算子筛查和KG独立复验')
    parser.add_argument('--offline', action='store_true', help='仅使用已下载的完整证据，不连接KGS或发起设备评测')
    parser.add_argument('--replay-only', action='store_true', help='仅报告本次历史方法复现，不混合旧HEAD的四版本合同')
    args = parser.parse_args()
    if args.replay_only:
        replay_report(args.operator, args.campaign_batch or args.batch)
        return
    finalize(args.operator, args.batch, args.prior_closure, offline=args.offline,
             campaign_batch=args.campaign_batch)


if __name__ == '__main__':
    main()
