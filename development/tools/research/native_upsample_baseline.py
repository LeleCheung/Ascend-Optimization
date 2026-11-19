"""Measure untouched upstream through its native entrypoint; preserve raw failures."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT=Path('/data/hanle/ascend-optimization')
RUN=ROOT/'runs/native-upsample-contrast-20261118'
OLD=ROOT/'runs/night-research-20261116'


def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


def main():
    sys.path.insert(0,str(ROOT/'Ascend-Optimization/hgkda/src'))
    from hgkda.adapters.kgs import call, upload_bundle
    port=19728;server=f'http://127.0.0.1:{port}'
    sock=socket.socket()
    assert sock.connect_ex(('127.0.0.1',port))!=0,'port already in use'
    sock.close()
    old=json.loads((OLD/'service.json').read_bytes())
    shell=old['command'][-1].replace(str(OLD),str(RUN)).replace('19718',str(port))
    deps=ROOT/'Ascend-Optimization/development/dependencies/third_party/FlagGems/src'
    shell=shell.replace('export PYTHONPATH=','export PYTHONPATH='+str(deps)+':')
    command=['docker','exec','-w',str(RUN),'tle_yy','bash','-c',shell]
    with (RUN/'kgs.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(RUN/'service.json',dict(pid=child.pid,command=command,server=server,timing='profiler',scope='project-owned'))
    for attempt in range(60):
        try:
            status=call(server,'/status');break
        except Exception:
            if child.poll() is not None:raise RuntimeError((RUN/'kgs.log').read_text()[-3000:])
            time.sleep(2)
    else:raise TimeoutError('KGS start')
    save(RUN/'service-status.json',status)
    archive=OLD/'no-simulator/operator.tar'
    (RUN/'operator.tar').write_bytes(archive.read_bytes());upload_bundle(server,RUN/'operator.tar')
    request=json.loads((OLD/'no-simulator/request.json').read_bytes())
    raw=(RUN/'upstream.py').read_bytes()
    request['implementation'].update(name='native_flaggems_upsample_d6a8eec',entrypoint='main.py::run',sources=[{'path':'main.py','content':'from upstream_native import upsample_bilinear2d as run\n'},{'path':'upstream_native.py','content':raw.decode()}])
    save(RUN/'native-request.json',request)
    save(RUN/'native-submission.json',dict(source_sha256=hashlib.sha256(raw).hexdigest(),entrypoint=request['implementation']['entrypoint'],algorithm_modified=False,reference='frozen CANN layout compatibility',timing='profiler',state='submitting'))
    import fcntl
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result=call(server,'/evaluate',request,timeout=2100,operation_id='native-upsample-20261118-round0')
    save(RUN/'native-result.json',result)
    print(json.dumps({'result':str(RUN/'native-result.json'),'status':result.get('status')}),flush=True)


if __name__=='__main__':main()
