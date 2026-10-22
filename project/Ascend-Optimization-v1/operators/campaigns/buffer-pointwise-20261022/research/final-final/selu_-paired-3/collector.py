#!/usr/bin/env python3
"""适配本机CANN的成对采集；调用者持项目卡锁，无旧目录删除。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def collect(spec_path, output, groups):
    if output.exists():
        raise ValueError('采集目录已存在，请使用新目录')
    output.mkdir(parents=True)
    driver_source = Path(__file__).with_name('paired-driver.py')
    driver = output/'driver.py'
    driver.write_bytes(driver_source.read_bytes())
    (output/'collector.py').write_bytes(Path(__file__).read_bytes())
    spec = json.loads(spec_path.read_text())
    source = Path(spec['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != spec['source_sha256']:
        raise ValueError('待采集源码与spec不同')
    (output/'spec.json').write_bytes(spec_path.read_bytes())
    help_text = subprocess.run(['msprof', 'op', '--help'], capture_output=True, text=True).stdout
    (output/'msprof-help.txt').write_text(help_text)
    if groups not in help_text:
        raise ValueError('本机帮助未声明所选指标组：'+groups)
    for side in ['reference', 'candidate']:
        directory = output/side
        directory.mkdir()
        app = shlex.join([sys.executable, str(driver), '--spec', str(spec_path), '--side', side])
        discovery = ['msprof', 'op', '--application='+app,
            '--output='+str(directory/'discovery'), '--kernel-name=__hgkda_no_match__', '--launch-count=1']
        result = subprocess.run(discovery, capture_output=True, text=True, timeout=300)
        log = result.stdout+'\n'+result.stderr
        (directory/'discovery.log').write_text(log)
        names = list(dict.fromkeys(re.findall(r'Kernel (.+?) skipped: not selected via --kernel-name\.', log)))
        # 输入构建也会发起Cast/Copy等kernel，只选目标运算；未知时保留日志并停止。
        tokens = ['log10', 'log_'] if spec['operator'] == 'log10_' else ['selu', 'elu_']
        selected = [name for name in names if any(token in name.lower() for token in tokens)]
        save(directory/'discovery.json', dict(command=discovery, returncode=result.returncode,
            names=names, selected=selected))
        if result.returncode or len(selected) != 1:
            raise RuntimeError('目标kernel不唯一：'+side+' '+repr(selected))
        command = ['msprof', 'op', '--application='+app,
            '--output='+str(directory/'profile'), '--kernel-name='+selected[0],
            '--launch-count=1', '--warm-up=5', '--aic-metrics='+groups]
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
        profile_log = result.stdout+'\n'+result.stderr
        (directory/'profile.log').write_text(profile_log)
        basic = list((directory/'profile').rglob('OpBasicInfo.csv'))
        complete = ('Op profiling finish.' in profile_log and 'Profiling data parse finished.' in profile_log)
        save(directory/'collection.json', dict(command=command, returncode=result.returncode,
            expected_kernel=selected[0], groups=groups, basic_info=[str(x) for x in basic],
            side=side, spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
            source_sha256=spec['source_sha256'], completion_marker=complete,
            status='completed' if result.returncode == 0 and len(basic) == 1 and complete else 'needs_review'))
        if result.returncode or len(basic) != 1 or not complete:
            raise RuntimeError('采集未完整结束，保留原始文件：'+str(directory))
    print('成对采集完成：'+str(output), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--groups', default='Roofline')
    a = p.parse_args()
    collect(a.spec, a.output, a.groups)
