#!/usr/bin/env python3
"""复用分项测量脚本，仅以已修复的设备任务计时替代空名称解析。"""
import importlib.util
from pathlib import Path

import torch
from triton.backends.ascend import testing


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    helper = load('copy_timer', 'ascend-copy-timer.py')
    original = load('scopes', 'probe-current-timing.py')
    # 只修改当前诊断模块的引用，包本身与其他任务不变。
    original.do_bench_npu = helper.make_device_task_timer(testing)
    original.main()


if __name__ == '__main__':
    main()
