#!/usr/bin/env python3
"""统一四算子的最佳源码复现与历史建议驱动优化，保存分开的结果。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

from optimization_campaign import PROJECT, checked, digest, save

spec = importlib.util.spec_from_file_location('campaign_verifier', Path(__file__).with_name('verify-agent-campaign.py'))
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def registry():
    items = json.loads(Path(__file__).with_name('four-operators.json').read_text(encoding='utf-8'))
    for item in items:
        for kind in ('best', 'method_seed'):
            source = PROJECT / item[kind]['source']
            if digest(source.read_text(encoding='utf-8')) != item[kind]['sha256']:
                raise ValueError(f'{item["operator"]} {kind}源码SHA不匹配')
        expected = ('19657', 'device_task') if item['operator'] == 'narrow_copy' else ('19655', 'device_kernel')
        if (str(item['port']), item['timing_scope']) != expected:
            raise ValueError('KGS与计时范围不匹配')
    return items


def campaign_path(item, batch, mode):
    return PROJECT / 'operators' / item['operator'] / 'reports' / f'profiling-agent-{batch}-{mode}'


def summarize(items, batch, modes, *, replay_batch=None):
    rows = []
    for item in items:
        for mode in modes:
            source_batch = replay_batch if mode == 'replay' and replay_batch else batch
            root = campaign_path(item, source_batch, mode)
            state_path = root / 'campaign-state.json'
            if not state_path.exists():
                continue
            state = json.loads(state_path.read_text(encoding='utf-8'))
            row = {'operator': item['operator'], 'mode': mode, 'status': state['status'],
                   'batch': source_batch,
                   'timing_scope': item['timing_scope'], 'campaign': root.relative_to(PROJECT).as_posix()}
            if state['status'] == 'completed':
                verification = verifier.verify(root)
                seed, seed_score = checked(root / 'seed-evaluation/seed-1.result.json')
                expected_seed = item['best' if mode == 'replay' else 'method_seed']['sha256']
                if seed['source_sha256'] != expected_seed:
                    raise ValueError('批次起点与四算子配置不匹配')
                if len(seed['rows']) != item['full_tests'] or seed['timing_scope'] != item['timing_scope']:
                    raise ValueError('完整测试范围或计时范围改变')
                if mode == 'replay' and state['final_source_sha256'] != item['best']['sha256']:
                    raise ValueError('最佳版本复现改变了历史源码')
                row.update(verification, seed_score=seed_score, final_score=min(state['final_scores']),
                           relative_seed=min(state['final_scores']) / seed_score,
                           generated_rounds=sum('source_sha256' in r and r['status'] != 'REPLAYED'
                                                for r in state['history']),
                           accepted_rounds=[r['round'] for r in state['history'] if r.get('accepted')],
                           improved_source=state['final_source_sha256'] != state['seed_sha256'])
                row['method_effective'] = (mode == 'method' and row['improved_source']
                                           and row['relative_seed'] > 1.005)
            rows.append(row)
    save(PROJECT / 'operators' / f'四算子闭环-{batch}.json', rows)
    lines = ['# 四算子工作流闭环', '', f'批次：`{batch}`。最佳源码复现与建议驱动优化分别记录。', '',
             '| 算子 | 模式 | 完整测试 | 起点 | 最终两次，相对PyTorch | 较慢值 | 相对起点 | 建议产生有效优化 |',
             '| --- | --- | ---: | ---: | --- | ---: | ---: | --- |']
    for row in rows:
        if row['status'] != 'VERIFIED':
            lines.append(f'| {row["operator"]} | {row["mode"]} | {row["status"]} | — | — | — | — | — |')
            continue
        mode = '最佳源码复现' if row['mode'] == 'replay' else '建议驱动优化'
        scores = ' / '.join(f'{s:.3f}×' for s in row['final_scores'])
        effective = '—' if row['mode'] == 'replay' else ('是' if row['method_effective'] else '本批未证实')
        lines.append(f'| {row["operator"]} | {mode} | {row["full_tests"]}/{row["full_tests"]} | '
                     f'{row["seed_score"]:.3f}× | {scores} | {row["final_score"]:.3f}× | '
                     f'{row["relative_seed"]:.3f}× | {effective} |')
    lines += ['', 'narrow_copy使用device_task，包含DMA和kernel；其他三个使用device_kernel。性能case取几何平均，最终采用两次完整复验较慢值。', '',
              '方法起点由四算子配置明确固定：logaddexp2从master出发，其余三个从历史中间版本出发，分别验证后续有效建议。', '',
              '方法有效须有模型生成的新源码、采纳记录及最终复验相对起点超过0.5%；固定源码复验不计为重新生成优化。数值Roofline尚未建立。', '',
              '完整测试、源码SHA、原始附件与最佳选择已由verify-agent-campaign.py核验；失败和未采纳轮次保留在各批次。', '']
    if replay_batch and replay_batch != batch:
        lines += [f'最佳复现来自`{replay_batch}`，建议驱动优化来自`{batch}`。各列为各自批次实测；方法改善只比较该方法批次内的起点与最终，不将跨HEAD结果当成同合同消融。', '']
    for row in rows:
        lines.append(f'- `{row["operator"]}` / `{row["mode"]}`：[证据目录]({row["campaign"].removeprefix("operators/")}/campaign-state.json)。')
    (PROJECT / 'operators' / f'四算子闭环-{batch}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--mode', choices=('all', 'replay', 'method'), default='all')
    parser.add_argument('--replay-batch', help='汇总已有最佳复现批次，避免修复生成接口后重复设备复验')
    parser.add_argument('--operators', nargs='+', choices=('logaddexp2', 'amin', 'matmul_bias_activation', 'narrow_copy'))
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--report-only', action='store_true', help='离线核验与汇总，不调用设备或模型')
    args = parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+', args.batch) or not 1 <= args.rounds <= 6:
        parser.error('批次名只允许字母数字横线下划线，轮数1—6')
    if args.replay_batch and not re.fullmatch('[A-Za-z0-9_-]+', args.replay_batch):
        parser.error('复现批次名只允许字母数字横线下划线')
    items = registry()
    if args.operators:
        items = [item for item in items if item['operator'] in args.operators]
    modes = ('replay', 'method') if args.mode == 'all' else (args.mode,)
    report_modes = ('replay', 'method') if args.replay_batch else modes
    if args.report_only:
        print(json.dumps(summarize(items, args.batch, report_modes, replay_batch=args.replay_batch), ensure_ascii=False, indent=2))
        return
    for mode in modes:
        for item in items:
            source_batch = args.replay_batch if mode == 'replay' and args.replay_batch else args.batch
            root = campaign_path(item, source_batch, mode)
            command = [sys.executable, str(Path(__file__).with_name('optimization_campaign.py')),
                       '--operator', item['operator'], '--seed', str(PROJECT / item['best' if mode == 'replay' else 'method_seed']['source']),
                       '--output', str(root), '--server', f'http://127.0.0.1:{item["port"]}',
                       '--timing-scope', item['timing_scope'], '--rounds', str(1 if mode == 'replay' else args.rounds),
                       '--profile-cases', '2', '--knowledge', str(Path(__file__).with_name('工作流通用知识.md')),
                       '--plain-json', '--max-output-tokens', '16384']
            if mode == 'replay':
                command += ['--thinking-budget', '8192', '--replay']
            else:
                command += ['--provider-thinking', 'disabled']
            log = root.parent / f'workflow-{args.batch}-{mode}.log'
            print('WORKFLOW_START', item['operator'], mode, str(log), flush=True)
            with log.open('ab') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            summarize(items, args.batch, report_modes, replay_batch=args.replay_batch)
            if result.returncode:
                raise RuntimeError(f'{item["operator"]} / {mode}停止，见{log}')
            print('WORKFLOW_COMPLETED', item['operator'], mode, flush=True)
    print('FOUR_OPERATOR_WORKFLOW_COMPLETED', args.batch, flush=True)


if __name__ == '__main__':
    main()
