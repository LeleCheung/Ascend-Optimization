"""v4用双缓冲队列建立MTE2/Vector/MTE3依赖；避免每块PIPE_ALL全流水栅栏。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3];ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'

POINTWISE=r'''
template<typename T,int OP>
__aicore__ inline void Pointwise(GM_ADDR x, GM_ADDR y, uint32_t n) {
    using namespace AscendC;
    constexpr uint32_t TILE=4096;
    TPipe pipe;
    TQue<QuePosition::VECIN,2> input;
    TQue<QuePosition::VECOUT,2> output;
    TBuf<TPosition::VECCALC> fbuffer,tmpbuffer,auxbuffer,maskbuffer;
    pipe.InitBuffer(input,2,TILE*sizeof(T));pipe.InitBuffer(output,2,TILE*sizeof(T));
    if constexpr (!__is_same(T,float)) pipe.InitBuffer(fbuffer,TILE*4);
    if constexpr (OP==4 || OP==5) pipe.InitBuffer(tmpbuffer,TILE*4);
    if constexpr (OP==5) {pipe.InitBuffer(auxbuffer,TILE*4);pipe.InitBuffer(maskbuffer,TILE/8);}
    LocalTensor<float> tmp,aux;LocalTensor<uint8_t> mask;
    if constexpr (OP==4 || OP==5) tmp=tmpbuffer.Get<float>();
    if constexpr (OP==5) {aux=auxbuffer.Get<float>();mask=maskbuffer.Get<uint8_t>();}
    GlobalTensor<T> src,dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),n);dst.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),n);
    for(uint32_t start=GetBlockIdx()*TILE;start<n;start+=GetBlockNum()*TILE) {
        uint32_t count=n-start<TILE?n-start:TILE;
        auto incoming=input.AllocTensor<T>();DataCopy(incoming,src[start],count);input.EnQue(incoming);
        auto raw=input.DeQue<T>();auto result=output.AllocTensor<T>();
        LocalTensor<float> values;
        if constexpr (__is_same(T,float)) values=raw.template ReinterpretCast<float>();
        else {values=fbuffer.Get<float>();Cast(values,raw,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();}
        Compute<OP>(values,tmp,aux,mask,count);PipeBarrier<PIPE_V>();
        if constexpr (__is_same(T,float)) DataCopy(result,values,count);
        else Cast(result,values,RoundMode::CAST_RINT,count);
        output.EnQue(result);input.FreeTensor(raw);
        auto outgoing=output.DeQue<T>();DataCopy(dst[start],outgoing,count);output.FreeTensor(outgoing);
    }
}
'''

FMOD=r'''
template<typename T>
__aicore__ inline void Fmod(GM_ADDR a,GM_ADDR b,uint32_t n) {
    using namespace AscendC;
    constexpr uint32_t TILE=4096;
    TPipe pipe;
    TQue<QuePosition::VECIN,2> inputA,inputB;
    TQue<QuePosition::VECOUT,2> output;
    TBuf<TPosition::VECCALC> fa,fb,tmp;
    pipe.InitBuffer(inputA,2,TILE*sizeof(T));pipe.InitBuffer(inputB,2,TILE*sizeof(T));
    pipe.InitBuffer(output,2,TILE*sizeof(T));pipe.InitBuffer(tmp,TILE*4);
    if constexpr (!__is_same(T,float)) {pipe.InitBuffer(fa,TILE*4);pipe.InitBuffer(fb,TILE*4);}
    auto q=tmp.Get<float>();
    GlobalTensor<T> ga,gb;ga.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a),n);gb.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b),n);
    for(uint32_t start=GetBlockIdx()*TILE;start<n;start+=GetBlockNum()*TILE) {
        uint32_t count=n-start<TILE?n-start:TILE;
        auto inA=inputA.AllocTensor<T>();auto inB=inputB.AllocTensor<T>();
        DataCopy(inA,ga[start],count);DataCopy(inB,gb[start],count);inputA.EnQue(inA);inputB.EnQue(inB);
        auto rawA=inputA.DeQue<T>();auto rawB=inputB.DeQue<T>();auto result=output.AllocTensor<T>();
        LocalTensor<float> x,y;
        if constexpr (__is_same(T,float)) {x=rawA.template ReinterpretCast<float>();y=rawB.template ReinterpretCast<float>();}
        else {x=fa.Get<float>();y=fb.Get<float>();Cast(x,rawA,RoundMode::CAST_NONE,count);Cast(y,rawB,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();}
        Div(q,x,y,count);PipeBarrier<PIPE_V>();Cast(q,q,RoundMode::CAST_TRUNC,count);PipeBarrier<PIPE_V>();
        Mul(q,q,y,count);PipeBarrier<PIPE_V>();Sub(x,x,q,count);PipeBarrier<PIPE_V>();
        if constexpr (__is_same(T,float)) DataCopy(result,x,count);
        else Cast(result,x,RoundMode::CAST_RINT,count);
        output.EnQue(result);inputA.FreeTensor(rawA);inputB.FreeTensor(rawB);
        auto outgoing=output.DeQue<T>();DataCopy(ga[start],outgoing,count);output.FreeTensor(outgoing);
    }
}
'''


def main():
    folder=BATCH/'native-build-v4';folder.mkdir(exist_ok=True)
    if '--host' not in sys.argv:
        previous=BATCH/'native-build-v3';source=(previous/'pointwise.cpp').read_text()
        begin=source.index('template<typename T,int OP>');end=source.index('#define KERNELS(',begin)
        source=source[:begin]+POINTWISE+'\n'+source[end:]
        begin=source.index('template<typename T>\n__aicore__ inline void Fmod(');end=source.index('extern "C" __global__ __aicore__ void hgkda_fmod_f32',begin)
        source=source[:begin]+FMOD+'\n'+source[end:]
        (folder/'pointwise.cpp').write_text(source,encoding='utf8');(folder/'CMakeLists.txt').write_bytes((previous/'CMakeLists.txt').read_bytes())
        host=(previous/'ascendc_host.cpp').read_text().replace('(n+16383)/16384','(n+4095)/4096')
        (folder/'ascendc_host.cpp').write_text(host,encoding='utf8')
        build=ROOT/'runtime/remaining-nine-ascendc-build-v4';build.mkdir(parents=True,exist_ok=True)
        command='set -euo pipefail\nsource /usr/local/Ascend/cann-9.0.0/set_env.sh\ncmake -S '+str(folder)+' -B '+str(build)+' -DCMAKE_BUILD_TYPE=Release\ncmake --build '+str(build)+' -j1\n'
        with (build/'build.log').open('ab') as log:
            result=subprocess.run(['docker','exec','-i','tle_yy','bash','-s'],input=command.encode(),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('v4计算构建失败 '+str(build/'build.log'))
        binary=build/'lib/libhgkda_remaining.so'
        record=dict(status='built',binary=str(binary),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'pointwise.cpp',folder/'CMakeLists.txt',binary]})
        target=BATCH/'ascendc-build-v4.json'
    else:
        os.environ['MAX_JOBS']='1'
        import torch,torch_npu
        from torch.utils.cpp_extension import load
        npu=Path(torch_npu.__file__).parent;host=ROOT/'runtime/remaining-nine-host-build-v4';host.mkdir(parents=True,exist_ok=True)
        source=folder/'ascendc_host.cpp'
        module=load(name='_hgkda_remaining_host_v4',sources=[str(source)],build_directory=str(host),
            extra_include_paths=[str(npu/'include'),'/usr/local/Ascend/cann-9.0.0/include'],extra_cflags=['-O2','-g0'],
            extra_ldflags=['-L'+str(npu/'lib'),'-ltorch_npu','-Wl,-rpath,'+str(npu/'lib')],with_cuda=False,verbose=True)
        binary=Path(module.__file__)
        record=dict(status='built',binary=str(binary),torch=torch.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,
            files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,binary]})
        target=BATCH/'host-build-v4.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(record))


if __name__=='__main__':main()
