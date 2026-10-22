#!/usr/bin/env python3
"""等待本批缓冲候选终态，将结果随知识覆盖层提供给CC，再执行追加优化。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('/data/hanle/ascend-optimization')
    tools = Path(__file__).resolve().parent
    batch = tools.parents[2]/'operators/campaigns/buffer-pointwise-20261022'
    # 使用计划中的固定项目路径，避免控制脚本移动后意外操作其他批次。
    assert batch.is_dir(), batch
    start = time.monotonic()
    while True:
        paths = [batch/('candidate-state-'+suffix+'.json') for suffix in ['ubsave-v1', 'ubsave-v2']]
        states = [json.loads(p.read_text()) if p.exists() else {} for p in paths]
        if any(d.get('status') == 'needs_attention' for d in states):
            raise RuntimeError('独立候选控制器异常，先检查，不启动追加设备任务')
        if all(d.get('status') == 'completed' for d in states):
            break
        if time.monotonic()-start > 10800:
            raise RuntimeError('等待候选超时，保留已有任务')
        time.sleep(10)
    evidence_path = batch/'paired-evidence.json'
    evidence = json.loads(evidence_path.read_text())
    if 'buffer_probes' in evidence:
        raise RuntimeError('本批证据已冻结，不重复生成')
    evidence['buffer_probes'] = []
    for state, path in zip(states, paths):
        for op, result in state['results'].items():
            directory = batch/'results'/op/('codex-'+path.stem[len('candidate-state-'):])
            dense_path = directory/'dense-semantics.json'
            dense = json.loads(dense_path.read_text()) if dense_path.exists() else None
            if dense:
                dense = dict(status=dense['status'], passed=dense['passed'], total=dense['total'],
                    source_sha256=dense['source_sha256'], evidence=str(dense_path),
                    failures=[{k: (str(v)[-1800:] if k == 'error' else v)
                               for k, v in case.items()} for case in dense['cases']
                              if case['status'] != 'passed'])
            evidence['buffer_probes'].append(dict(operator=op, state_path=str(path),
                status=result.get('status'), scores=result.get('scores'),
                dense=dense))
    raw = (json.dumps(evidence, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    if len(raw) > 128*1024:
        raise RuntimeError('诊断摘要过大，需精简后再启动')
    evidence_path.write_bytes(raw)
    provenance = dict(evidence_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        contribution='Codex汇总缓冲选项实验事实，冻结入HG-KDA角色；CC/DeepSeek继续生成')
    (batch/'kg-launch-provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
    subprocess.run([sys.executable, str(tools/'run-paired-kg.py'), '--batch', batch.name,
                    '--seed-plan', '--dense', '--rounds', '3'], check=True)


if __name__ == '__main__':
    main()
