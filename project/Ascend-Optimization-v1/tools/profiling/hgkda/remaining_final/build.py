"""独立构建大分块、无片上输出复制的候选，历史二进制不覆盖。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3];ROOT=PROJECT.parents[2]
OLD=PROJECT/'operators/campaigns/remaining-nine-20261030'
BATCH=PROJECT/'operators/campaigns/remaining-final-20261031'

POINTWISE=r'''
template<typename T,int OP>
__aicore__ inline void Pointwise(GM_ADDR x, GM_ADDR y, uint32_t n) {
    using namespace AscendC;
    constexpr uint32_t TILE=OP==5?8192:16384;
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuffer,floatBuffer,tmpBuffer,auxBuffer,maskBuffer;
    pipe.InitBuffer(rawBuffer,2*TILE*sizeof(T));
    if constexpr (!__is_same(T,float)) pipe.InitBuffer(floatBuffer,TILE*4);
    if constexpr (OP==4 || OP==5) pipe.InitBuffer(tmpBuffer,TILE*4);
    if constexpr (OP==5) {pipe.InitBuffer(auxBuffer,TILE*4);pipe.InitBuffer(maskBuffer,TILE/8);}
    auto rawBase=rawBuffer.Get<T>();
    LocalTensor<float> tmp,aux;LocalTensor<uint8_t> mask;
    if constexpr (OP==4 || OP==5) tmp=tmpBuffer.Get<float>();
    if constexpr (OP==5) {aux=auxBuffer.Get<float>();mask=maskBuffer.Get<uint8_t>();}
    GlobalTensor<T> src,dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),n);dst.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),n);
    uint32_t iteration=0;
    bool used[2]={false,false};
    for(uint32_t start=GetBlockIdx()*TILE;start<n;start+=GetBlockNum()*TILE,++iteration) {
        uint32_t count=n-start<TILE?n-start:TILE;
        uint32_t slot=iteration&1;
        event_t event=static_cast<event_t>(slot);
        if(used[slot]) WaitFlag<HardEvent::MTE3_MTE2>(event);
        auto raw=rawBase[slot*TILE];
        DataCopy(raw,src[start],count);
        SetFlag<HardEvent::MTE2_V>(event);WaitFlag<HardEvent::MTE2_V>(event);
        LocalTensor<float> values;
        if constexpr (__is_same(T,float)) values=raw.template ReinterpretCast<float>();
        else {values=floatBuffer.Get<float>();Cast(values,raw,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();}
        Compute<OP>(values,tmp,aux,mask,count);PipeBarrier<PIPE_V>();
        if constexpr (!__is_same(T,float)) {Cast(raw,values,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();}
        SetFlag<HardEvent::V_MTE3>(event);WaitFlag<HardEvent::V_MTE3>(event);
        DataCopy(dst[start],raw,count);
        SetFlag<HardEvent::MTE3_MTE2>(event);used[slot]=true;
    }
    for(uint32_t slot=0;slot<2;slot++) if(used[slot]) WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(slot));
}
'''

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['event-large-v6','sync-large-v7','prefetch-large-v8','contiguous-cores-v10'],default='event-large-v6');a=parser.parse_args()
    folder=BATCH/('native-build-'+a.variant);folder.mkdir(parents=True,exist_ok=True)
    manifest=BATCH/('device-build-'+a.variant+'.json')
    if manifest.exists():raise FileExistsError(manifest)
    source=(OLD/'native-build-v4/pointwise.cpp').read_text()
    if a.variant in {'event-large-v6','prefetch-large-v8','contiguous-cores-v10'}:
        begin=source.index('template<typename T,int OP>');end=source.index('#define KERNELS(',begin)
        pointwise=POINTWISE
        if a.variant in {'prefetch-large-v8','contiguous-cores-v10'}:
            pointwise=pointwise.replace('    for(uint32_t start=GetBlockIdx()*TILE;', '''    uint32_t first=GetBlockIdx()*TILE;
    uint32_t step=GetBlockNum()*TILE;
    if(first<n) {
        uint32_t firstCount=n-first<TILE?n-first:TILE;
        DataCopy(rawBase,src[first],firstCount);SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
    }
    for(uint32_t start=GetBlockIdx()*TILE;''')
            pointwise=pointwise.replace('''        if(used[slot]) WaitFlag<HardEvent::MTE3_MTE2>(event);
        auto raw=rawBase[slot*TILE];
        DataCopy(raw,src[start],count);
        SetFlag<HardEvent::MTE2_V>(event);WaitFlag<HardEvent::MTE2_V>(event);''', '''        auto raw=rawBase[slot*TILE];
        WaitFlag<HardEvent::MTE2_V>(event);
        uint32_t next=start+step;
        if(next<n) {
            uint32_t nextSlot=slot^1;event_t nextEvent=static_cast<event_t>(nextSlot);
            if(used[nextSlot]) WaitFlag<HardEvent::MTE3_MTE2>(nextEvent);
            uint32_t nextCount=n-next<TILE?n-next:TILE;
            auto nextRaw=rawBase[nextSlot*TILE];
            DataCopy(nextRaw,src[next],nextCount);SetFlag<HardEvent::MTE2_V>(nextEvent);
        }''')
        if a.variant=='contiguous-cores-v10':
            pointwise=pointwise.replace('''    uint32_t first=GetBlockIdx()*TILE;
    uint32_t step=GetBlockNum()*TILE;''','''    uint32_t blocks=(n+TILE-1)/TILE;
    uint32_t tilesPerCore=(blocks+GetBlockNum()-1)/GetBlockNum();
    uint32_t first=GetBlockIdx()*tilesPerCore*TILE;
    uint32_t end=first+tilesPerCore*TILE;
    end=end<n?end:n;
    uint32_t step=TILE;''')
            pointwise=pointwise.replace('for(uint32_t start=GetBlockIdx()*TILE;start<n;start+=GetBlockNum()*TILE,++iteration)', 'for(uint32_t start=first;start<end;start+=step,++iteration)')
            pointwise=pointwise.replace('if(next<n)', 'if(next<end)')
        source=source[:begin]+pointwise+'\n'+source[end:]
    else:
        original=(OLD/'native-build-v3/pointwise.cpp').read_text()
        begin=original.index('template<typename T,int OP>');end=original.index('#define KERNELS(',begin)
        block=original[begin:end]
        begin=source.index('template<typename T,int OP>');end=source.index('#define KERNELS(',begin)
        source=source[:begin]+block+'\n'+source[end:]
    # 启动核数仍由原C++封装决定；无工作核安全返回，不改变输入合同。
    src=folder/'pointwise.cpp';src.write_text(source,encoding='utf8')
    shutil=__import__('shutil');shutil.copyfile(OLD/'native-build-v4/CMakeLists.txt',folder/'CMakeLists.txt')
    build=ROOT/'runtime'/('remaining-final-'+a.variant);build.mkdir(parents=True,exist_ok=True)
    command='set -euo pipefail\nsource /usr/local/Ascend/cann-9.0.0/set_env.sh\ncmake -S '+str(folder)+' -B '+str(build)+' -DCMAKE_BUILD_TYPE=Release\ncmake --build '+str(build)+' -j1\n'
    with (build/'build.log').open('ab') as log:
        r=subprocess.run(['docker','exec','-i','tle_yy','bash','-s'],input=command.encode(),stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('构建失败 '+str(build/'build.log'))
    binary=build/'lib/libhgkda_remaining.so'
    value=dict(status='built',variant=a.variant,binary=str(binary),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [src,folder/'CMakeLists.txt',binary]})
    manifest.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(value))

if __name__=='__main__':main()
