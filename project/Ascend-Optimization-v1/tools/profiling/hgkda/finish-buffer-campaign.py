#!/usr/bin/env python3
"""串联本批CC终态、父源码对照和最终成对验证；可由tmux独立托管。"""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time


def main():
    tools = Path(__file__).resolve().parent
    batch = tools.parents[2]/'operators/campaigns/buffer-pointwise-20261022'
    started = time.monotonic()
    while True:
        path = batch/'kg-paired-state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get('status') == 'needs_attention':
            raise RuntimeError('KG控制器异常，先检查，不推进新设备任务')
        if state.get('status') == 'completed':
            break
        if time.monotonic()-started > 18000:
            raise RuntimeError('等待KG超时；不终止或重复提交原任务')
        time.sleep(10)
    # 第三轮BF16路径缩小了指数范围；第二轮仅host优化的独立源码也进入门禁。
    host_source = (Path('/data/hanle/ascend-optimization/runs')/batch.name/'kg-runs'/
        ('selu_-'+batch.name+'-paired-hgkda')/'stages/optimize/work/.kernelgen/evals/round-0002/main.py')
    host_raw = host_source.read_bytes()
    if hashlib.sha256(host_raw).hexdigest() != '9ac565b2a89fc4f4da3c38bfd231cf2da46e57705344d596d51ac27a34c27aa8':
        raise ValueError('SELU第二轮源码不同，先审查')
    host_target = batch/'candidates/selu_-host-v5.py'
    if host_target.exists():raise FileExistsError(host_target)
    host_target.write_bytes(host_raw)
    (batch/'host-v5-provenance.json').write_text(json.dumps(dict(source=str(host_target),
        original=str(host_source),source_sha256=hashlib.sha256(host_raw).hexdigest(),
        contribution='CC第二轮生成的纯host优化；Codex单独保留，避免被更高分但数值范围错误的轮次遮盖'),
        ensure_ascii=False,indent=2)+'\n')
    for op in ['log10_', 'selu_']:
        source = batch/'seeds'/(op+'.py')
        target = batch/'candidates'/(op+'-parent.py')
        if target.exists():
            raise FileExistsError('父源码对照已存在，不重复启动')
        target.write_bytes(source.read_bytes())
    # Codex独立补测：子向量绑定粒度与select分析分别记账，不归为CC生成。
    subprocess.run([sys.executable, str(tools/'run-paired-candidates.py'), '--batch', batch.name,
                    '--suffix', 'mechanism-v3', '--dense'], check=True)
    subprocess.run([sys.executable, str(tools/'run-paired-candidates.py'), '--batch', batch.name,
                    '--suffix', 'pipeline-v4', '--operators', 'log10_', '--dense'], check=True)
    subprocess.run([sys.executable, str(tools/'run-paired-candidates.py'), '--batch', batch.name,
                    '--suffix', 'host-v5', '--operators', 'selu_', '--dense'], check=True)
    # 新批次同环境测父源码，用真实延迟判断是否改进；每份源码仅一次正式评测。
    subprocess.run([sys.executable, str(tools/'run-paired-candidates.py'), '--batch', batch.name,
                    '--suffix', 'parent', '--dense'], check=True)
    subprocess.run([sys.executable, str(tools/'finish-paired.py'), '--batch', batch.name,
        '--suffix', 'final', '--wait-suffixes', 'parent', '--wait-kg', '--skip-backfill',
        '--arms', 'codex-parent', 'paired-HG-KDA', 'codex-ubsave-v1', 'codex-ubsave-v2',
        'codex-mechanism-v3', 'codex-pipeline-v4', 'codex-host-v5'], check=True)
    # 离线验收用容器Python3.11；宿主Python3.7不运行新语法的导出/比较模块。
    subprocess.run(['docker', 'exec', 'tle_yy', '/usr/local/python3.11.15/bin/python3.11',
        str(tools/'audit-paired.py'), '--batch-dir', str(batch), '--tag', 'final',
        '--output', str(batch/'completion-audit.json')], check=True)


if __name__ == '__main__':
    main()
