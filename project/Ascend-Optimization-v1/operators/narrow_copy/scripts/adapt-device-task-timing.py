#!/usr/bin/env python3
"""在新的 case API 副本修复 DMA 的 N/A 名称解析，算子与计时计算不变。"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

MASTER = 'd6a8eec473517a3d68157b208eb9c057eb1d4c50'
BENCHMARK = 'benchmark/test_narrow_copy.py'
SOURCE = 'src/flag_gems/ops/narrow_copy.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--provenance', type=Path, required=True)
    args = parser.parse_args()
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(args.checkout), *arguments])
    git('merge-base', '--is-ancestor', MASTER, 'HEAD')
    if git('status', '--porcelain=v1').strip():
        raise SystemExit('独立副本不干净，拒绝覆盖')
    source = git('show', MASTER + ':' + SOURCE)
    assert (args.checkout / SOURCE).read_bytes() == source
    path = args.checkout / BENCHMARK
    original = path.read_bytes()
    text = original.decode('utf-8')
    assert 'case_fn=narrow_copy_case_fn,' in text
    anchor = 'class TensorSelectBenchmark(base.GenericBenchmark2DOnly):\n'
    assert text.count(anchor) == 1 and 'base.Config.mode =' not in text
    text = text.replace(anchor, anchor + '''    def _time_callable(self, fn, xs):
        if base.vendor_name == "ascend" and base.Config.mode == base.consts.BenchMode.KERNEL:
            from .ascend_copy_timer import time_device_tasks
            return time_device_tasks(fn)
        return super()._time_callable(fn, xs)

''')
    compile(text, BENCHMARK, 'exec')
    adapted = text.encode('utf-8')
    helper = Path(__file__).with_name('ascend-copy-timer.py').read_bytes()
    path.write_bytes(adapted)
    (args.checkout / 'benchmark/ascend_copy_timer.py').write_bytes(helper)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps({
        'upstream_commit': MASTER,
        'parent_checkout_commit': git('rev-parse', 'HEAD').decode().strip(),
        'benchmark_path': BENCHMARK,
        'original_benchmark_sha256': hashlib.sha256(original).hexdigest(),
        'adapted_benchmark_sha256': hashlib.sha256(adapted).hexdigest(),
        'timer_helper_sha256': hashlib.sha256(helper).hexdigest(),
        'operator_path': SOURCE,
        'operator_sha256': hashlib.sha256(source).hexdigest(),
        'timing_scope': 'device_task',
        'change': '只将 kernel_name 转为 pandas string 后再匹配；保留原 task_time 耗时、采样数及聚合，包括 DMA 与 AIV kernel',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('NARROW_DEVICE_TASK_TIMING_ADAPTED')


if __name__ == '__main__':
    main()
