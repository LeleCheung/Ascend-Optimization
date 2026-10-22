#!/usr/bin/env python3
"""等待v3终态后，构建和独立评测无输入缓存的SELU分块整合版。"""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    tools=Path(__file__).resolve().parent
    s=importlib.util.spec_from_file_location('selu_v4_queue',tools/'run-nightly.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    q=m.Queue('paired-pointwise-20261022',Path('/data/hanle/ascend-optimization'),'tle_yy')
    deadline=time.monotonic()+14400
    while True:
        p=q.batch/'candidate-state-profilemeta-v3.json'
        if p.exists():
            state=json.loads(p.read_text())
            if state['status']=='needs_attention':raise RuntimeError('v3需先处理')
            if state['status']=='completed':break
        if time.monotonic()>deadline:raise RuntimeError('等待超预算')
        time.sleep(10)
    q.run('prepare-selu-v4',' '.join(shlex.quote(str(x)) for x in [q.py,tools/'prepare-selu-v4.py']),600)
    subprocess.run(['python3',str(tools/'run-paired-candidates.py'),'--suffix','profilemeta-v4',
                    '--operators','selu_','--dense'],check=True)


if __name__=='__main__':main()
