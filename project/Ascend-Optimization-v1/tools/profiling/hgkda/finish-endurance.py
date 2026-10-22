#!/usr/bin/env python3
"""等待连续队列完成后离线验收和生成中文汇报；不提交NPU工作。"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True)
    a=p.parse_args()
    if not a.batch.replace('-','').replace('_','').isalnum():raise ValueError('非法批次名')
    tools=Path(__file__).resolve().parent
    batch=tools.parents[2]/'operators/campaigns'/a.batch
    deadline=time.monotonic()+14*3600
    while True:
        state=batch/'endurance-state.json'
        if state.exists() and json.loads(state.read_text())['status']=='completed':break
        if time.monotonic()>deadline:raise TimeoutError('等待队列超时；原任务保留，不重发设备工作')
        time.sleep(15)
    output=batch/'completion-audit.json'
    for script,args in [('audit-endurance.py',['--batch-dir',batch,'--output',output]),
                        ('summarize-endurance.py',['--batch-dir',batch])]:
        if script.startswith('audit') and output.exists():continue
        subprocess.run(['docker','exec','tle_yy','/usr/local/python3.11.15/bin/python3.11',str(tools/script)]+[str(v) for v in args],check=True)


if __name__=='__main__':main()
