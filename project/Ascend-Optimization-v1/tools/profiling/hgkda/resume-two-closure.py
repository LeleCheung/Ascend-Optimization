#!/usr/bin/env python3
"""确认预算取消已完成、服务空闲后，保存恢复凭据并续跑两算子队列。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request

TOOLS = Path(__file__).resolve().parent
ROOT = Path('/data/hanle/ascend-optimization')


def read(path):
    return json.loads(path.read_text())


def execute(batch_name, host_probe):
    assert batch_name == 'two-closure-20261024', '本恢复工具只适用于已审计批次'
    spec = importlib.util.spec_from_file_location('two_resume', TOOLS / 'run-two-closure.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batch = ROOT / 'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns' / batch_name
    workspace = ROOT / 'runs' / batch_name / 'kg-runs' / ('upsample_bilinear2d-' + batch_name + '-attempt-1')
    with (ROOT / 'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = read(workspace / '.kernelgen/run-process.json')
        assert metadata['state'] == 'EXITED' and metadata['finished_at']
        assert not read(workspace / '.kernelgen/active-server-operations.json')['operations']
        campaign = module.Campaign(batch_name)
        campaign.q.idle()
        recovery = batch / 'recovery-after-budget.json'
        assert not recovery.exists(), '恢复已有记录，先核查当前任务身份'
        before = read(batch / 'endurance-state.json')
        assert before['status'] == 'needs_attention'
        record = dict(reason='预算取消已完成，三个服务确认空闲；保留旧状态并续跑',
                      previous_state=before, worker=metadata)
        # 保存取消时的完整状态，历史证据不覆盖。
        recovery.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        if host_probe:
            folder = batch / 'research/host-components-standalone'
            folder.mkdir(exist_ok=False)
            source = (batch / 'knowledge/probe-cann-host-components.py').read_text()
            request = dict(command=['{python}', 'probe.py'], files=[dict(path='probe.py', content=source)], timeout_seconds=300)
            (folder / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n')
            response = json.load(campaign.q.opener.open(urllib.request.Request(
                campaign.q.server + '/debug/jobs', data=json.dumps(request).encode(),
                headers={'Content-Type': 'application/json'}, method='POST'), timeout=345))
            (folder / 'response.json').write_text(json.dumps(response, ensure_ascii=False, indent=2) + '\n')
            campaign.q.idle()
        # 独立验收的新候选优先级按真实延迟确定，避免继续使用较慢的libentry种子。
        rows = read(batch / 'results/upsample_bilinear2d/codex-late/summary.json')['records']
        selected = next(row for row in rows if row['name'] == 'safe_compiled_launch' and row['status'] == 'passed')
        source = Path(selected['source'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == selected['source_sha256']
        formal = read(Path(selected['result']))
        assert formal['status'] == 'PASSED' and formal['num_passed'] == formal['num_workloads'] == 24
        gates = [source.parent / name for name in ['remaining-semantics.json', 'special.json']]
        for path in gates:
            gate = read(path)
            assert gate['status'].lower() == 'passed' and gate['source_sha256'] == selected['source_sha256']
        current = campaign.state['operators']['upsample_bilinear2d']['best']
        assert selected['latency_ms'] < current['latency_ms']
        campaign.state['operators']['upsample_bilinear2d']['best'] = dict(selected, gates=[str(path) for path in gates])
        campaign.state.pop('error', None)
        campaign.save()
        try:
            campaign.execute()
        except Exception as error:
            campaign.state.update(status='needs_attention', error=type(error).__name__ + ': ' + str(error))
            campaign.save()
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--host-probe', action='store_true')
    args = parser.parse_args()
    execute(args.batch, args.host_probe)
