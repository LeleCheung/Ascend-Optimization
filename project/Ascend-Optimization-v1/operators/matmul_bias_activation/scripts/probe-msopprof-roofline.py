#!/usr/bin/env python3
"""通过 KGS debug 队列验证当前 910B/CANN 能否产出官方 Roofline。"""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys


def main():
    root = Path('/data/hanle/ascend-optimization/goal-20261010/FlagGems')
    source = Path('flaggems-master-20261010.py').resolve()
    artifacts = Path(os.environ['KGS_DEBUG_ARTIFACTS'])
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != 'd6a8eec473517a3d68157b208eb9c057eb1d4c50':
        raise RuntimeError('FlagGems 提交不匹配')
    case = 'benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::2'
    command = [
        'msprof', 'op', '--kernel-name=matmul_bias_activation_kernel_mix_aic',
        '--warm-up=1', '--launch-count=1', '--aic-metrics=Roofline',
        '--replay-mode=kernel', '--output=' + str(artifacts / 'roofline'),
        sys.executable, '-m', 'kernelgen_server.profiling.gems_runner',
        '--backend', 'npu', '--case-id', case,
        '--completion-marker', str(artifacts / 'runner-completed'), '--',
        '-q', 'benchmark/test_matmul_bias_activation.py', '-m', 'matmul_bias_activation',
        '--level', 'core', '--profile-only', '--case-id', case,
        '--profile-warmup', '1', '--profile-iterations', '2',
        '--override', 'matmul_bias_activation:' + str(source) + ':run',
    ]
    provenance = {'case_id': case, 'flaggems_revision': revision,
                  'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'command': command, 'purpose': '验证官方 Roofline 原始采集，不是正式加速比评测'}
    (artifacts / 'request.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    env = dict(os.environ)
    env['GEMS_VENDOR'] = 'ascend'
    env['PYTHONPATH'] = os.pathsep.join([str(source.parent), str(root / 'src'), str(root), env.get('PYTHONPATH', '')])
    # 由 KGS debug 的 timeout_seconds=300 约束整个进程组；不另建进程组。
    # 直接写原始输出，超时也保留已经产生的证据。
    with (artifacts / 'stdout.txt').open('w', encoding='utf-8') as stdout, \
         (artifacts / 'stderr.txt').open('w', encoding='utf-8') as stderr:
        result = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
    code = result.returncode
    files = [str(p.relative_to(artifacts)) for p in (artifacts / 'roofline').rglob('*')
             if p.is_file() and p.suffix == '.csv']
    raw_stdout = (artifacts / 'stdout.txt').read_text(encoding='utf-8', errors='replace')
    bound = re.search(r'latency bound:\s*([^\r\n]+)', raw_stdout)
    summary = {'exit_code': code, 'roofline_files': files,
               'official_latency_bound': bound.group(1).strip() if bound else None,
               'status': 'raw_files_available' if code == 0 and files else 'no_usable_roofline_files',
               'next_step': '校验字段、kernel/range 与 roof 来源后再接入分析；成功退出本身不代表有有效 Roofline'}
    (artifacts / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
