"""只读核查本项目服务、卡锁和已结束队列；不提交设备工作。"""
import fcntl
import json
from pathlib import Path
import urllib.request

ROOT=Path('/data/hanle/ascend-optimization')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
services={}
for port in [19655,19657,19659]:
    value=json.load(opener.open('http://127.0.0.1:%d/status'%port,timeout=15))
    services[str(port)]=value.get('scheduler')
with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        available=True
    except BlockingIOError:
        available=False
state=ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/next25-20261029-v2/next-state.json'
old=json.loads(state.read_text())
print(json.dumps(dict(services=services,project_lock_available=available,
                     previous_queue_status=old.get('status'),previous_current_job=old.get('current_job')),ensure_ascii=False))
