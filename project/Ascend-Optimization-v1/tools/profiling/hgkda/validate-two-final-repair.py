#!/usr/bin/env python3
"""主队列结束后持卡锁验证上采样的分块写入归属修复，并采集最终机制证据。"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path('/data/hanle/ascend-optimization')
TOOLS = Path(__file__).resolve().parent
BATCH = ROOT / 'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024'


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def execute():
    output = BATCH / 'research/disjoint-validation-control'
    output.mkdir(exist_ok=False)
    end = time.monotonic() + 2400
    save(output / 'control.json', dict(status='waiting_for_main_completion'))
    while read(BATCH / 'endurance-state.json')['status'] != 'completed':
        state = read(BATCH / 'endurance-state.json')
        if state['status'] == 'needs_attention':
            raise RuntimeError('主队列需先恢复，未提交新设备任务')
        if time.monotonic() > end:
            raise TimeoutError('未等到主队列完成，未提交新设备任务')
        time.sleep(5)
    with (ROOT / 'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in (ROOT / 'runs' / BATCH.name / 'kg-runs').glob('*/.kernelgen/run-process.json'):
            assert read(path).get('finished_at')
            active = path.with_name('active-server-operations.json')
            if active.exists():
                assert not read(active)['operations']
        spec = importlib.util.spec_from_file_location('disjoint_validation_queue', TOOLS / 'run-nightly.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        q = module.Queue(BATCH.name, ROOT, 'tle_yy')
        q.idle()
        save(output / 'control.json', dict(status='validating_disjoint_candidate'))
        process = subprocess.run([sys.executable, str(TOOLS / 'validate-two-disjoint.py')],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1500)
        save(output / 'validation-process.json', dict(returncode=process.returncode,
             stdout=process.stdout.decode(errors='replace'), stderr=process.stderr.decode(errors='replace')))
        q.idle()
        assert process.returncode == 0
        row = read(BATCH / 'results/upsample_bilinear2d/codex-disjoint/summary.json')['records'][0]
        assert row['status'] == 'passed', row
        source = Path(row['source'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_sha256']
        # 从快路径的输入行归属推导输出行，每行必须恰有一个program写入。
        coverage = []
        for height in [4, 8, 16, 32, 64, 128, 256]:
            for block in [1, 2, 4, 8, 16, 32]:
                if height % block or block > height // 2:
                    continue
                counts = [0] * (2 * height)
                for b in range(height // block):
                    low = max(b * block - 1, 0)
                    for k in range(block):
                        if b == 0 and k == block - 1:
                            continue
                        counts[2 * (low + k) + 1] += 1
                        counts[2 * (low + k) + 2] += 1
                counts[0] += 1
                counts[-1] += 1
                assert counts == [1] * (2 * height)
                coverage.append(dict(height=height, block=block, unique_row_owner=True))
        save(output / 'write-ownership.json', dict(status='passed', source_sha256=row['source_sha256'],
             previous_sha256='3008f880a9b9ab1da020736990abf52e47b0a281bb3f42115f8e87bf08d82c83',
             reason='原首块和下一块在交界处重复写一对输出行；修复后首块不写最后一对，所有输出行归属唯一', cases=coverage))
        def run(key, arguments, budget):
            try:
                q.run(key, ' '.join(shlex.quote(str(v)) for v in arguments), budget)
            except RuntimeError:
                path = BATCH / 'research/upsample-final-reviewed-metrics/response.json'
                if key != 'metrics-ups-disjoint-final' or not path.exists():
                    raise
                response = read(path)
                if response['status'].lower() != 'failed' or 'completion marker' not in json.dumps(response).lower():
                    raise
                q.idle()
                return dict(status='failed', response=str(path), reason='msprof未生成完成标记；保留失败，成对trace单列')
            return dict(status='completed')
        project = BATCH.parents[2]
        run('observe-ups-disjoint-final', [q.py, TOOLS / 'observe-endurance.py', '--operator', 'upsample_bilinear2d',
            '--source', source, '--leaf', BATCH / 'catalog/ops/upsample_bilinear2d',
            '--output', BATCH / 'research/upsample-final-reviewed', '--index', '1'], 1200)
        result = Path(row['result'])
        raw = read(result)
        worst = max((r for r in raw['per_workload'] if r['phase'] == 'timing'), key=lambda r: r['latency_ms'])
        metrics = run('metrics-ups-disjoint-final', [q.py, project / 'tools/evaluation/profile-operator.py',
            '--evaluation', result, '--request', result.with_name(result.name.replace('.result.json', '.request.json')),
            '--inspect', result.parent / 'inspect.json', '--case-id', worst['uuid'],
            '--output', BATCH / 'research/upsample-final-reviewed-metrics', '--server', q.server,
            '--level', 'metrics', '--aic-metrics', 'PipeUtilization', '--timeout', '240'], 420)
        q.idle()
        save(output / 'control.json', dict(status='completed' if metrics['status']=='completed' else 'completed_with_metrics_gap',
             candidate=row, metrics=metrics,
             ownership=str(output / 'write-ownership.json'), scope='仅本批卡7；未暂停其他进程'))
        print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    execute()
