"""独立v3构建：加入保留Div/Trunc/Mul/Sub顺序的自写Ascend C余数。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3];ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'

FMOD=r'''
template<typename T>
__aicore__ inline void Fmod(GM_ADDR a,GM_ADDR b,uint32_t n) {
    using namespace AscendC;
    constexpr uint32_t TILE=8192;
    TPipe pipe;
    TBuf<TPosition::VECCALC> ra,rb,fa,fb,tmp;
    pipe.InitBuffer(ra,TILE*sizeof(T));pipe.InitBuffer(rb,TILE*sizeof(T));
    if constexpr (!__is_same(T,float)) {pipe.InitBuffer(fa,TILE*4);pipe.InitBuffer(fb,TILE*4);}
    pipe.InitBuffer(tmp,TILE*4);
    auto rawA=ra.Get<T>();auto rawB=rb.Get<T>();
    LocalTensor<float> x,y;
    if constexpr (__is_same(T,float)) {x=ra.Get<float>();y=rb.Get<float>();}
    else {x=fa.Get<float>();y=fb.Get<float>();}
    auto q=tmp.Get<float>();
    GlobalTensor<T> ga,gb;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a),n);
    gb.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b),n);
    for(uint32_t start=GetBlockIdx()*TILE;start<n;start+=GetBlockNum()*TILE) {
        uint32_t count=n-start<TILE?n-start:TILE;
        DataCopy(rawA,ga[start],count);DataCopy(rawB,gb[start],count);PipeBarrier<PIPE_ALL>();
        if constexpr (!__is_same(T,float)) {
            Cast(x,rawA,RoundMode::CAST_NONE,count);Cast(y,rawB,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        }
        Div(q,x,y,count);PipeBarrier<PIPE_V>();
        Cast(q,q,RoundMode::CAST_TRUNC,count);PipeBarrier<PIPE_V>();
        Mul(q,q,y,count);PipeBarrier<PIPE_V>();
        Sub(x,x,q,count);PipeBarrier<PIPE_V>();
        if constexpr (!__is_same(T,float)) {Cast(rawA,x,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();}
        PipeBarrier<PIPE_ALL>();DataCopy(ga[start],rawA,count);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" __global__ __aicore__ void hgkda_fmod_f32(GM_ADDR a,GM_ADDR b,uint32_t n){Fmod<float>(a,b,n);}
extern "C" __global__ __aicore__ void hgkda_fmod_f16(GM_ADDR a,GM_ADDR b,uint32_t n){Fmod<half>(a,b,n);}
extern "C" __global__ __aicore__ void hgkda_fmod_bf16(GM_ADDR a,GM_ADDR b,uint32_t n){Fmod<bfloat16_t>(a,b,n);}
extern "C" void hgkda_fmod_launch(uint32_t blocks,void* stream,uint8_t* a,uint8_t* b,uint32_t n,int dtype) {
    if(dtype==0) hgkda_fmod_f32<<<blocks,nullptr,stream>>>(a,b,n);
    else if(dtype==1) hgkda_fmod_f16<<<blocks,nullptr,stream>>>(a,b,n);
    else if(dtype==27) hgkda_fmod_bf16<<<blocks,nullptr,stream>>>(a,b,n);
}
'''


def main():
    folder=BATCH/'native-build-v3';folder.mkdir(exist_ok=True)
    if '--host' not in sys.argv:
        previous=BATCH/'native-build-v2'
        (folder/'pointwise.cpp').write_text((previous/'pointwise.cpp').read_text()+FMOD,encoding='utf8')
        (folder/'CMakeLists.txt').write_bytes((previous/'CMakeLists.txt').read_bytes())
        host=(previous/'ascendc_host.cpp').read_text()
        host=host.replace('kernels[6]','kernels[7]').replace('"hgkda_expm1_launch"','"hgkda_expm1_launch","hgkda_fmod_launch"').replace('i<6','i<7').replace('op<6','op<7')
        host=host.replace('if(!eligible) return supplied ? fallback(object,output) : fallback(object);',
            'if(op==6 && !supplied) eligible=false;\n    if(!eligible) return supplied ? fallback(object,output) : fallback(object);')
        host=host.replace('return supplied?output:', 'return op==6?object:(supplied?output:').replace('THPVariable_Wrap(out)));','THPVariable_Wrap(out))));')
        (folder/'ascendc_host.cpp').write_text(host,encoding='utf8')
        build=ROOT/'runtime/remaining-nine-ascendc-build-v3';build.mkdir(parents=True,exist_ok=True)
        command='set -euo pipefail\nsource /usr/local/Ascend/cann-9.0.0/set_env.sh\ncmake -S '+str(folder)+' -B '+str(build)+' -DCMAKE_BUILD_TYPE=Release\ncmake --build '+str(build)+' -j1\n'
        with (build/'build.log').open('ab') as log:
            result=subprocess.run(['docker','exec','-i','tle_yy','bash','-s'],input=command.encode(),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('v3计算构建失败 '+str(build/'build.log'))
        binary=build/'lib/libhgkda_remaining.so'
        record=dict(status='built',binary=str(binary),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'pointwise.cpp',folder/'CMakeLists.txt',binary]})
        target=BATCH/'ascendc-build-v3.json'
    else:
        os.environ['MAX_JOBS']='1'
        import torch,torch_npu
        from torch.utils.cpp_extension import load
        npu=Path(torch_npu.__file__).parent
        host=ROOT/'runtime/remaining-nine-host-build-v3';host.mkdir(parents=True,exist_ok=True)
        source=folder/'ascendc_host.cpp'
        module=load(name='_hgkda_remaining_host_v3',sources=[str(source)],build_directory=str(host),
            extra_include_paths=[str(npu/'include'),'/usr/local/Ascend/cann-9.0.0/include'],extra_cflags=['-O2','-g0'],
            extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib')],with_cuda=False,verbose=True)
        binary=Path(module.__file__)
        record=dict(status='built',binary=str(binary),torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,
            files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,binary]})
        target=BATCH/'host-build-v3.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(record))


if __name__=='__main__':main()
