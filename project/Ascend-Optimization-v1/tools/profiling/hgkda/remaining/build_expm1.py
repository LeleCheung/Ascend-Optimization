"""独立生成v2构建目录，加入保留小值精度的Ascend C expm1；不覆盖v1附件。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'


def main():
    folder=BATCH/'native-build-v2';folder.mkdir(exist_ok=True)
    if '--host' not in sys.argv:
        source=(HERE/'ascendc/pointwise.cpp').read_text()
        source=source.replace('AscendC::LocalTensor<float> tmp, uint32_t count)',
            'AscendC::LocalTensor<float> tmp, AscendC::LocalTensor<float> aux, AscendC::LocalTensor<uint8_t> mask, uint32_t count)')
        source=source.replace('} else if constexpr (OP==4) {','''} else if constexpr (OP==5) {
        Exp(tmp,value,count);PipeBarrier<PIPE_V>();
        Adds(tmp,tmp,-1.0f,count);PipeBarrier<PIPE_V>();
        Abs(aux,value,count);PipeBarrier<PIPE_V>();
        CompareScalar(mask,aux,0.01f,CMPMODE::LT,count);PipeBarrier<PIPE_V>();
        Muls(aux,value,0.5f,count);PipeBarrier<PIPE_V>();
        Adds(aux,aux,1.0f,count);PipeBarrier<PIPE_V>();
        Mul(aux,value,aux,count);PipeBarrier<PIPE_V>();
        Select(value,mask,aux,tmp,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);
    } else if constexpr (OP==4) {''')
        source=source.replace('constexpr uint32_t TILE=16384;','constexpr uint32_t TILE=OP==5 ? 8192 : 16384;')
        source=source.replace('rawBuffer,floatBuffer,tmpBuffer;','rawBuffer,floatBuffer,tmpBuffer,auxBuffer,maskBuffer;')
        source=source.replace('if constexpr (OP==4) pipe.InitBuffer(tmpBuffer,TILE*sizeof(float));',
            'if constexpr (OP==4 || OP==5) pipe.InitBuffer(tmpBuffer,TILE*sizeof(float));\n    if constexpr (OP==5) {pipe.InitBuffer(auxBuffer,TILE*sizeof(float));pipe.InitBuffer(maskBuffer,TILE/8);}')
        source=source.replace('if constexpr (OP==4) tmp=tmpBuffer.Get<float>();',
            'if constexpr (OP==4 || OP==5) tmp=tmpBuffer.Get<float>();\n    LocalTensor<float> aux;LocalTensor<uint8_t> mask;\n    if constexpr (OP==5) {aux=auxBuffer.Get<float>();mask=maskBuffer.Get<uint8_t>();}')
        source=source.replace('Compute<OP>(values,tmp,count)','Compute<OP>(values,tmp,aux,mask,count)')
        source+='\nKERNELS(5,hgkda_expm1)\n'
        (folder/'pointwise.cpp').write_text(source,encoding='utf8')
        (folder/'CMakeLists.txt').write_bytes((HERE/'ascendc/CMakeLists.txt').read_bytes())
        host=(HERE/'ascendc_host.cpp').read_text()
        host=host.replace('kernels[5]','kernels[6]').replace('"hgkda_hardswish_launch"','"hgkda_hardswish_launch","hgkda_expm1_launch"').replace('i<5','i<6').replace('op<5','op<6').replace('op<2?','(op<2 || op==5)?')
        (folder/'ascendc_host.cpp').write_text(host,encoding='utf8')
        build=ROOT/'runtime/remaining-nine-ascendc-build-v2';build.mkdir(parents=True,exist_ok=True)
        command='set -euo pipefail\nsource /usr/local/Ascend/cann-9.0.0/set_env.sh\ncmake -S '+str(folder)+' -B '+str(build)+' -DCMAKE_BUILD_TYPE=Release\ncmake --build '+str(build)+' -j1\n'
        with (build/'build.log').open('ab') as log:
            result=subprocess.run(['docker','exec','-i','tle_yy','bash','-s'],input=command.encode(),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('v2计算构建失败 '+str(build/'build.log'))
        binary=build/'lib/libhgkda_remaining.so'
        record=dict(status='built',binary=str(binary),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'pointwise.cpp',folder/'CMakeLists.txt',binary]})
        target=BATCH/'ascendc-build-v2.json'
    else:
        os.environ['MAX_JOBS']='1'
        import torch,torch_npu
        from torch.utils.cpp_extension import load
        npu=Path(torch_npu.__file__).parent
        host=ROOT/'runtime/remaining-nine-host-build-v2';host.mkdir(parents=True,exist_ok=True)
        source=folder/'ascendc_host.cpp'
        module=load(name='_hgkda_remaining_host_v2',sources=[str(source)],build_directory=str(host),
            extra_include_paths=[str(npu/'include'),'/usr/local/Ascend/cann-9.0.0/include'],extra_cflags=['-O2','-g0'],
            extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib')],with_cuda=False,verbose=True)
        binary=Path(module.__file__)
        record=dict(status='built',binary=str(binary),torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,
            files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,binary]})
        target=BATCH/'host-build-v2.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(record))


if __name__=='__main__':main()
