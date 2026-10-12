#!/usr/bin/env python3
"""从仓库固定源码生成 narrow_copy 评测副本，校验与已归档计时合同一致。"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PROJECT = REPO / 'project/Ascend-Optimization-v1'


def prepare(repo, target):
    repo = repo.resolve()
    source = repo / 'third_party/FlagGems'
    manifest = json.loads((repo / 'manifests/flaggems-master-files.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path = source / row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('固定源码 SHA 不一致：' + row['path'])
    if target.exists():
        raise ValueError('评测副本已存在，请核验或选用新目录：' + str(target))
    scripts = repo / 'project/Ascend-Optimization-v1/operators/narrow_copy/scripts'
    evidence = repo / 'project/Ascend-Optimization-v1/operators/narrow_copy/reports/device-timing-contract-20261010'
    text = (source / 'benchmark/test_narrow_copy.py').read_text(encoding='utf-8')
    anchor = '\ndef narrow_copy_gbps(bench_fn_args, latency):'
    functions = '''
def narrow_copy_case_fn(shape, cur_dtype):
    dim = 0
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"dim": dim, "start": shape[dim] // 4, "length": shape[dim] // 2},
        builder_args=(tuple(shape),),
    )


def narrow_copy_build_inputs(plan, cur_dtype, device):
    return next(narrow_copy_input_fn(plan.builder_args[0], cur_dtype, device))

'''
    if text.count(anchor) != 1 or text.count('input_fn=narrow_copy_input_fn,') != 1:
        raise ValueError('固定 master 的 benchmark 结构不匹配')
    text = text.replace(anchor, '\n' + functions + anchor)
    text = text.replace('input_fn=narrow_copy_input_fn,',
                        'case_fn=narrow_copy_case_fn,\n        build_inputs_fn=narrow_copy_build_inputs,')
    anchor = 'class TensorSelectBenchmark(base.GenericBenchmark2DOnly):\n'
    if text.count(anchor) != 1:
        raise ValueError('找不到唯一的 benchmark 类')
    text = text.replace(anchor, anchor + '''    def _time_callable(self, fn, xs):
        if base.vendor_name == "ascend" and base.Config.mode == base.consts.BenchMode.KERNEL:
            from .ascend_copy_timer import time_device_tasks
            return time_device_tasks(fn)
        return super()._time_callable(fn, xs)

''')
    adapted = text.encode('utf-8')
    helper = (scripts / 'ascend-copy-timer.py').read_bytes()
    expected = {
        'benchmark/test_narrow_copy.py': adapted,
        'benchmark/ascend_copy_timer.py': helper,
    }
    for name, data in expected.items():
        if (evidence / name).read_bytes() != data:
            raise ValueError('生成内容与已归档的设备任务计时合同不一致：' + name)
    compile(text, 'benchmark/test_narrow_copy.py', 'exec')
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '.pytest_cache'))
    for name, data in expected.items():
        (target / name).write_bytes(data)
    result = {'upstream_commit': manifest['upstream_commit'], 'source': str(source),
              'timing_scope': 'device_task',
              'patched_files': {name: hashlib.sha256(data).hexdigest() for name, data in expected.items()}}
    (target / 'ascend-evaluation-adapter.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.repo, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
