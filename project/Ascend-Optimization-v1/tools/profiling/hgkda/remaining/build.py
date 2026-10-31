"""单CPU构建独立Ascend C计算及C++宿主封装，记录版本和二进制身份。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',action='store_true');a=parser.parse_args()
    build=ROOT/'runtime/remaining-nine-ascendc-build';build.mkdir(parents=True,exist_ok=True)
    if not a.host:
        command='set -euo pipefail\nsource /usr/local/Ascend/cann-9.0.0/set_env.sh\ncmake -S '+str(HERE/'ascendc')+' -B '+str(build)+' -DCMAKE_BUILD_TYPE=Release\ncmake --build '+str(build)+' -j1\n'
        log=build/'build.log'
        with log.open('ab') as stream:
            r=subprocess.run(['docker','exec','-i','tle_yy','bash','-s'],input=command.encode(),stdout=stream,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('计算构建失败 '+str(log))
        libraries=list(build.rglob('libhgkda_remaining.so'))
        if len(libraries)!=1:raise RuntimeError('计算库不唯一')
        record=dict(status='built',binary=str(libraries[0]),log=str(log),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'ascendc/pointwise.cpp',HERE/'ascendc/CMakeLists.txt',libraries[0]]})
        target=BATCH/'ascendc-build.json'
    else:
        os.environ['MAX_JOBS']='1'
        import torch,torch_npu
        from torch.utils.cpp_extension import load
        npu=Path(torch_npu.__file__).parent
        host=ROOT/'runtime/remaining-nine-host-build';host.mkdir(parents=True,exist_ok=True)
        source=HERE/'ascendc_host.cpp'
        module=load(name='_hgkda_remaining_host_v1',sources=[str(source)],build_directory=str(host),
            extra_include_paths=[str(npu/'include'),'/usr/local/Ascend/cann-9.0.0/include'],extra_cflags=['-O2','-g0'],
            extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib')],with_cuda=False,verbose=True)
        binary=Path(module.__file__)
        record=dict(status='built',binary=str(binary),torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,binary]})
        target=BATCH/'host-build.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(record))


if __name__=='__main__':main()
