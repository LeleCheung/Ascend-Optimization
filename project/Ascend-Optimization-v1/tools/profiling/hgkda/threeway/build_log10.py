"""单CPU构建本项目Ascend C候选；独立runtime目录，不运行NPU负载。"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/data/hanle/ascend-optimization')
PROJECT=ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1'
source=PROJECT/'tools/profiling/hgkda/threeway/ascendc'
build=ROOT/'runtime/threeway-ascendc-log10-build'
build.mkdir(parents=True,exist_ok=True)
command='set -euo pipefail\nsource /usr/local/Ascend/cann-9.0.0/set_env.sh\ncmake -S '+str(source)+' -B '+str(build)+' -DCMAKE_BUILD_TYPE=Release\ncmake --build '+str(build)+' -j1\n'
log=build/'build.log'
with log.open('ab') as stream:
    result=subprocess.run(['docker','exec','-i','tle_yy','bash','-s'],input=command.encode(),stdout=stream,stderr=subprocess.STDOUT)
libraries=list(build.rglob('libhgkda_log10.so'))
manifest=dict(status='built' if result.returncode==0 and libraries else 'build_failed',returncode=result.returncode,
              files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [source/'log10.cpp',source/'CMakeLists.txt',*libraries]},log=str(log))
path=PROJECT/'operators/campaigns/threeway-20261030/ascendc-build.json'
if path.exists():
    raw=path.read_bytes()
    path.with_name('ascendc-build-history-'+hashlib.sha256(raw).hexdigest()[:12]+'.json').write_bytes(raw)
path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(manifest,ensure_ascii=False))
raise SystemExit(0 if manifest['status']=='built' else result.returncode or 1)
