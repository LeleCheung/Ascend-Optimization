"""Submit the raw native module with a separate KGS name-only adapter."""
import json
from pathlib import Path
import sys
import fcntl

r=Path('/data/hanle/ascend-optimization/runs/native-upsample-contrast-20261118')
sys.path.insert(0,'/data/hanle/ascend-optimization/Ascend-Optimization/hgkda/src')
from hgkda.adapters.kgs import call
request=json.loads((r/'native-request.json').read_bytes())
raw=(r/'upstream.py').read_text()
request['implementation']['entrypoint']='main.py::run'
request['implementation']['sources']=[{'path':'main.py','content':'from upstream_native import upsample_bilinear2d as run\n'},{'path':'upstream_native.py','content':raw}]
(r/'native-request-v2.json').write_text(json.dumps(request,indent=2))
with (r.parent/'physical-card-7-campaign.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    result=call('http://127.0.0.1:19728','/evaluate',request,timeout=2100,operation_id='native-upsample-20261118-round0-v2')
(r/'native-result.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'result':str(r/'native-result.json'),'status':result.get('status')}))
