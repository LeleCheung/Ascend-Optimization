"""测量经验吞吐与流式复制参考；不将观测最大值称为硬件理论峰值。"""
import argparse,hashlib,json,os,statistics,time
from pathlib import Path
import torch
import torch_npu
from triton.backends.ascend.testing import do_bench_npu

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    a.output.mkdir(parents=True);records=[]
    for dt in ['float16','bfloat16','float32']:
        for size in [2048,4096]:
            row=dict(kind='gemm',dtype=dt,size=size,work=2*size**3);records.append(row)
            x=torch.randn((size,size),dtype=getattr(torch,dt),device='npu');y=torch.randn_like(x);out=torch.empty_like(x)
            def call():torch.mm(x,y,out=out)
            call();torch.npu.synchronize()
            values=[float(do_bench_npu(call,warmup=10,active=30)) for _ in range(3)]
            row.update(device_ms=values,median_ms=statistics.median(values),tflops=row['work']/statistics.median(values)/1e9)
            print(json.dumps(row),flush=True)
            del x,y,out
    for size in [134217728,268435456]:
        x=torch.ones(size,dtype=torch.float16,device='npu');out=torch.empty_like(x)
        def call():out.copy_(x)
        call();torch.npu.synchronize()
        values=[float(do_bench_npu(call,warmup=10,active=30)) for _ in range(3)]
        records.append(dict(kind='copy',dtype='float16',elements=size,logical_bytes=size*4,
            device_ms=values,median_ms=statistics.median(values),logical_gbps=size*4/statistics.median(values)/1e6,
            scope='输入与输出均超过L2；数值仍是有效逻辑读写吞吐，不冒充硬件计数带宽'))
        del x,out
    result=dict(status='completed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=records,
        scope='固定软件栈torch.mm/torch.copy的经验设备吞吐；预分配输出；非理论峰值')
    (a.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
