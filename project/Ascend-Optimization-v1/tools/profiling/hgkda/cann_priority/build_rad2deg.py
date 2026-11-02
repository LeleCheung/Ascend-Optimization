"""单CPU构建自写弧度转角度候选，构建阶段不执行NPU任务。"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
from common import BATCH,HERE,PROJECT,ROOT,read,save,sha,load,PY


def device():
    record=BATCH/'rad2deg-device-build-v1.json'
    if record.exists():
        for p,digest in read(record)['files'].items():assert sha(p)==digest
        return
    folder=BATCH/'native-rad2deg-v1';folder.mkdir(parents=True,exist_ok=True)
    source=folder/'rad2deg.cpp';source.write_bytes((HERE/'rad2deg_kernel.cpp').read_bytes())
    cmake=folder/'CMakeLists.txt'
    cmake.write_text('''cmake_minimum_required(VERSION 3.16)
project(hgkda_rad2deg LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "构建类型")
set(SOC_VERSION "Ascend910B4" CACHE STRING "目标SoC")
set(RUN_MODE "npu" CACHE STRING "执行模式")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-9.0.0" CACHE PATH "固定CANN")
include(${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(hgkda_rad2deg SHARED rad2deg.cpp)
''',encoding='utf8')
    build=ROOT/'runtime/cann-priority-rad2deg-device-v1';build.mkdir(parents=True,exist_ok=True)
    command='set -euo pipefail\nsource /usr/local/Ascend/cann-9.0.0/set_env.sh\ncmake -S '+str(folder)+' -B '+str(build)+' -DCMAKE_BUILD_TYPE=Release\ncmake --build '+str(build)+' -j1\n'
    log=build/'build.log'
    with log.open('ab') as stream:
        subprocess.run(['docker','exec','-i','tle_yy','bash','-s'],input=command.encode(),stdout=stream,stderr=subprocess.STDOUT,check=True)
    binary=folder/'libhgkda_rad2deg.so';shutil.copyfile(build/'lib/libhgkda_rad2deg.so',binary)
    save(record,dict(status='built',files={str(p):sha(p) for p in [source,cmake,binary]},binary=str(binary),build_log=str(log)))


def host():
    record=BATCH/'rad2deg-host-build-v1.json'
    if record.exists():
        for p,digest in read(record)['files'].items():assert sha(p)==digest
        return
    os.environ['MAX_JOBS']='1'
    import torch,torch_npu
    from torch.utils.cpp_extension import load as compile_host
    folder=BATCH/'native-rad2deg-v1';source=folder/'rad2deg_host.cpp';source.write_bytes((HERE/'rad2deg_host.cpp').read_bytes())
    build=ROOT/'runtime/cann-priority-rad2deg-host-v1';build.mkdir(parents=True,exist_ok=True)
    npu=Path(torch_npu.__file__).parent
    module=compile_host(name='_hgkda_rad2deg_host_v1',sources=[str(source)],build_directory=str(build),
        extra_include_paths=[str(npu/'include'),'/usr/local/Ascend/cann-9.0.0/include'],extra_cflags=['-O2','-g0'],
        extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-ldl','-Wl,-rpath,'+str(npu/'lib')],with_cuda=False,verbose=True)
    binary=folder/Path(module.__file__).name;shutil.copyfile(module.__file__,binary)
    save(record,dict(status='built',binary=str(binary),files={str(p):sha(p) for p in [source,binary]},
        torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI))


def proposal():
    device_info=read(BATCH/'rad2deg-device-build-v1.json');host_info=read(BATCH/'rad2deg-host-build-v1.json')
    for info in [device_info,host_info]:
        for p,digest in info['files'].items():assert sha(p)==digest
    out=BATCH/'proposals/rad2deg_/ascendc-streaming-v1';out.mkdir(parents=True,exist_ok=True)
    files={}
    for info in [device_info,host_info]:
        for p in info['files']:
            p=Path(p);target=out/p.name
            if target.exists():assert sha(target)==sha(p)
            else:shutil.copyfile(p,target)
            files[str(target)]=sha(target)
    parent=BATCH/'starting-points/rad2deg_.py'
    code=parent.read_text(encoding='utf8')+'\n\n_original_run=run\n'
    h=out/Path(host_info['binary']).name;d=out/Path(device_info['binary']).name
    code+='''import hashlib as _hashlib,importlib.util as _importlib
from pathlib import Path as _Path
'''
    code+='_host_path=_Path('+repr(str(h))+')\n_device_path=_Path('+repr(str(d))+')\n'
    code+='assert _hashlib.sha256(_host_path.read_bytes()).hexdigest()=='+repr(sha(h))+'\n'
    code+='assert _hashlib.sha256(_device_path.read_bytes()).hexdigest()=='+repr(sha(d))+'\n'
    code+='assert torch.__version__=='+repr(host_info['torch'])+' and torch._C._GLIBCXX_USE_CXX11_ABI=='+repr(host_info['abi'])+'\n'
    code+='_spec=_importlib.spec_from_file_location("_hgkda_rad2deg_host_v1",_host_path)\n_host=_importlib.module_from_spec(_spec);_spec.loader.exec_module(_host)\n_host.configure_library(str(_device_path))\n'
    code+='def run(A):\n    return _host.execute(A,_original_run)\n'
    source=out/'main.py'
    if source.exists():assert source.read_text(encoding='utf8')==code
    else:source.write_text(code,encoding='utf8')
    files[str(source)]=sha(source)
    attribution='Codex自写Ascend C常量乘法与同stream C++启动，原KG Triton路径处理非连续及未对齐布局'
    save(out/'manifest.json',dict(source_sha256=sha(source),files=files,device_build=device_info,host_build=host_info,attribution=attribution))
    save(out/'proposal.json',dict(operator='rad2deg_',id=out.name,source='main.py',source_sha256=sha(source),
        backend='AscendC_with_Triton_layout_fallback',attribution=attribution,
        prediction='当前CANN对照0.767；减少Triton宿主分派，连续对齐输入用8K流式块和读缓冲直接写回；原地别名、IEEE特殊值及两种参考均须验证'))
    print('RAD2DEG_PROPOSAL',sha(source))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','device','host','proposal'],default='all');a=p.parse_args()
    if a.phase in {'all','device'}:device()
    if a.phase=='all':
        queue=load('rad_build_queue',HERE.parent/'run-nightly.py').Queue(BATCH.name,ROOT,'tle_yy')
        queue.run('build-rad2deg-host',PY+' '+str(__file__)+' --phase host',1200)
        proposal()
    elif a.phase=='host':host()
    elif a.phase=='proposal':proposal()


if __name__=='__main__':main()
