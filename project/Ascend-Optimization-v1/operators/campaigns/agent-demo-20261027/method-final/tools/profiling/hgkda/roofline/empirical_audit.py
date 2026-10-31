"""少量保全经验参考的原task CSV并独立重算；不重复搜索峰值。"""
import argparse,csv,hashlib,json,math,os,statistics
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True)
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    import torch,torch_npu
    from triton.backends.ascend.testing import do_bench_npu_profiler
    cases=[dict(kind='gemm',dtype=dt,size=4096) for dt in ['float16','bfloat16','float32']]+[dict(kind='copy',dtype='float16',size=268435456)]
    records=[]
    for case in cases:
        dtype=getattr(torch,case['dtype']);size=case['size'];directory=out/(case['kind']+'-'+case['dtype'])
        if case['kind']=='gemm':
            x=torch.randn((size,size),device='npu',dtype=dtype);y=torch.randn_like(x);target=torch.empty_like(x)
            def fn():torch.mm(x,y,out=target)
            work=2*size**3
        else:
            x=torch.ones(size,device='npu',dtype=dtype);target=torch.empty_like(x)
            def fn():target.copy_(x)
            work=size*4
        value=float(do_bench_npu_profiler(fn,warmup=3,active=10,prof_dir=str(directory),keep_res=True))
        files=list(directory.rglob('task_time*.csv'))
        if len(files)!=1:raise ValueError('缺少唯一原始task CSV')
        rows=list(csv.DictReader(files[0].open(encoding='utf-8-sig',newline='')))[1:-1]
        if len(rows)!=13:raise ValueError('经验参考不是预期单任务路径')
        times=[float(r['task_time(us)']) for r in rows[3:]]
        if any(not math.isfinite(v) or v<=0 for v in times):raise ValueError('无效设备时间')
        independent=statistics.mean(times)/1000
        if not math.isclose(value,independent,rel_tol=1e-9):raise ValueError('独立CSV重算与工具不一致')
        records.append(dict(**case,work=work,device_ms=value,active_device_us=times,
            throughput=work/value/(1e9 if case['kind']=='gemm' else 1e6),unit='TFLOPs' if case['kind']=='gemm' else 'GB/s',
            csv_path=str(files[0]),csv_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(),
            kernel_names=sorted(set(r['kernel_name'] for r in rows))))
        print(json.dumps(records[-1]),flush=True)
        del x,target
        if case['kind']=='gemm':del y
    (out/'results.json').write_text(json.dumps(dict(status='completed',records=records,
        purpose='保全参考的原始计时并重算；不替换前批三次观测包络，不称为物理上界'),ensure_ascii=False,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':main()
