#!/usr/bin/env python3
"""在已适配 case API 的独立副本中，为 narrow_copy 固定完整算子计时。"""
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
    anchor = 'def test_narrow_copy_perf():\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor +
        '    # 连续复制通过 DMA 执行，kernel CSV 不覆盖该路径；两侧共用完整调用计时。\n'
        '    base.Config.mode = base.consts.BenchMode.OPERATOR\n')
    compile(text, BENCHMARK, 'exec')
    adapted = text.encode('utf-8')
    path.write_bytes(adapted)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps({
        'upstream_commit': MASTER,
        'parent_checkout_commit': git('rev-parse', 'HEAD').decode().strip(),
        'benchmark_path': BENCHMARK,
        'original_benchmark_sha256': hashlib.sha256(original).hexdigest(),
        'adapted_benchmark_sha256': hashlib.sha256(adapted).hexdigest(),
        'operator_path': SOURCE,
        'operator_sha256': hashlib.sha256(source).hexdigest(),
        'timing_scope': 'FlagGems core operator：批量调用前后同步的平均完整调用耗时',
        'change': '所有版本与 PyTorch 统一使用上游已有的 OPERATOR 计时；保留全部形状、精度及正确性用例',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('NARROW_OPERATOR_TIMING_ADAPTED')


if __name__ == '__main__':
    main()
