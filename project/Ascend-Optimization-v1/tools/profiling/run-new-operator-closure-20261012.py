#!/usr/bin/env python3
"""筛选完成后运行新算子的原生 KG 两组和独立 Agent 四版本闭环。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
EVAL = PROJECT / 'tools/evaluation'
sys.path.insert(0, str(EVAL))
from kg_workspace_state import exportable_state


def run(command):
    print('RUN', json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', default='20261012-agent')
    a = parser.parse_args()
    selection = PROJECT / 'operators' / ('next-operator-screen-' + a.batch + '.json')
    rows = json.loads(selection.read_text(encoding='utf-8'))
    selected = [r for r in rows if r.get('selected')]
    if len(selected) != 1:
        raise ValueError('没有唯一、通过正确性且低于0.8的正式新对象')
    row = selected[0]
    operator = row['operator']
    seed = Path(row['source'])
    root = PROJECT / 'operators' / operator
    py = sys.executable
    controller_python = '/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python3'
    run_root = Path('/data/hanle/ascend-optimization/runs') / a.batch / 'kg-runs'
    # 本轮原生 KG 和独立 Agent 使用相同模型、起点和最多六轮预算。
    os.environ['KG_MODEL'] = 'deepseek-flash'
    controller = '/data/hanle/ascend-optimization/runtime/kg-controller'
    os.environ['PATH'] = controller + ':' + os.environ['PATH']
    for arm in ('no-profile', 'profile'):
        suffix = 'kg-' + arm + '-' + a.batch
        workspace = run_root / (operator + '-' + suffix)
        progress = workspace / '.kernelgen/run-progress.json'
        if workspace.exists():
            process = workspace / '.kernelgen/run-process.json'
            if not process.exists() or not json.loads(process.read_text()).get('finished_at'):
                run([controller_python, str(EVAL / 'wait-kg-worker.py'), str(workspace)])
        else:
            try:
                run(['bash', str(EVAL / 'run-kg-operator-910b.sh'), operator, arm, suffix, '6', str(seed)])
            except subprocess.CalledProcessError:
                # 原生优化已完成但审核等待不等于内核失败；只导出做独立验证。
                exportable_state(workspace, allow_review_waiting=True)
        exportable_state(workspace, allow_review_waiting=True)
        exported = root / 'reports' / ('kg-' + arm + '-' + a.batch) / 'export'
        if not exported.exists():
            run([py, str(EVAL / 'export-kg-best.py'), str(workspace), operator, str(exported), '--allow-review-waiting'])
        independent = exported.parent / 'independent'
        from optimization_campaign import evaluate
        evaluate(operator, exported / 'main.py', independent, 'kg-' + arm,
                 'http://127.0.0.1:19655', repeats=2)
        profile_evidence = exported.parent / 'native-evidence'
        if not profile_evidence.exists():
            run([py, str(EVAL / 'archive-kg-evidence.py'), str(workspace), str(profile_evidence), '--allow-review-waiting'])
    campaign = root / 'reports' / ('profiling-agent-' + a.batch)
    run([py, str(PROJECT / 'tools/profiling/optimization_campaign.py'),
         '--operator', operator, '--seed', str(seed), '--output', str(campaign),
         '--rounds', '6', '--profile-cases', '2', '--knowledge', str(PROJECT / 'tools/profiling/910B优化知识.md')])
    state = json.loads((campaign / 'campaign-state.json').read_text(encoding='utf-8'))
    final = min(enumerate(state['final_scores'], 1), key=lambda item: item[1])[0]
    command = [py, str(EVAL / 'compare-operator-versions.py'), '--version', 'master', row['result']]
    for arm in ('no-profile', 'profile'):
        from optimization_campaign import checked
        scores = [(i, checked(root / 'reports' / ('kg-' + arm + '-' + a.batch)
                             / 'independent' / f'kg-{arm}-{i}.result.json')[1]) for i in (1, 2)]
        repeat = min(scores, key=lambda item: item[1])[0]
        command += ['--version', 'kg-' + arm,
                    str(root / 'reports' / ('kg-' + arm + '-' + a.batch) / 'independent' / f'kg-{arm}-{repeat}.result.json')]
    command += ['--version', 'agent', str(campaign / 'final-evaluation' / f'final-{final}.result.json'),
                '--output', str(root / 'reports' / ('master-closure-' + a.batch) / '版本对比.md')]
    run(command)
    run([py, str(PROJECT / 'tools/profiling/finalize-agent-campaign.py'),
         '--operator', operator, '--batch', a.batch])
    print('NEW_OPERATOR_FOUR_VERSION_COMPLETED', operator, state['final_scores'], flush=True)


if __name__ == '__main__':
    main()
