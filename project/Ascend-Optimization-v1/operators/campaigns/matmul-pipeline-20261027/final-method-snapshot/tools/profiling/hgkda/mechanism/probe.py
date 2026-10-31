"""同编译产物的调用路径干预；不将同步walltime解释为纯设备时间。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import time

from core import diagnose


def load(path):
    spec=importlib.util.spec_from_file_location('mechanism_probe_candidate',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def capture_call(function,value):
    import torch
    from triton import knobs
    from triton.runtime.jit import JITFunction
    from triton.runtime import driver
    records=[]
    original=JITFunction.run
    def wrapped(jit,*args,grid,warmup,**kwargs):
        compiled=original(jit,*args,grid=grid,warmup=warmup,**kwargs)
        if not warmup:
            options=dict(kwargs)
            options['debug']=options.get('debug',jit.debug) or knobs.runtime.debug
            from triton.runtime import jit as jm
            jm._flagprism.apply_compile_options(options)
            device=driver.active.get_current_device()
            binder=jit.device_caches[device][4]
            bound,_,_=binder(*args,**options)
            actual_grid=grid(bound) if callable(grid) else grid
            if hasattr(compiled,'result'):compiled=compiled.result()
            records.append(dict(jit=jit,compiled=compiled,args=args,kwargs=kwargs,bound=bound,grid=tuple(actual_grid)))
        return compiled
    JITFunction.run=wrapped
    try:
        result=function(value)
        torch.npu.synchronize()
    finally:JITFunction.run=original
    return records,result


def direct_call(records):
    from triton import knobs
    from triton.runtime import driver
    # 只重放当前probe捕获的参数；绝不将此闭包用于候选或其他输入。
    def call():
        for record in records:
            c=record['compiled'];g=record['grid']
            stream=driver.active.get_current_stream(driver.active.get_current_device())
            metadata=c.launch_metadata(g,stream,*record['bound'].values())
            c.run(*(tuple(g)+(1,)*(3-len(g))),stream,c.function,c.packed_metadata,metadata,
                  knobs.runtime.launch_enter_hook,knobs.runtime.launch_exit_hook,*record['bound'].values())
    return call


def paired_measure(functions,reset,trials=7,batch=12):
    import torch
    values={k:[] for k in functions}
    for round in range(trials):
        order=list(functions)
        if round%2:order.reverse()
        for key in order:
            reset();torch.npu.synchronize()
            begin=time.perf_counter_ns()
            for _ in range(batch):functions[key]()
            torch.npu.synchronize()
            values[key].append((time.perf_counter_ns()-begin)/1000/batch)
    return values


def probe(source,operator,output,shapes,trace=False):
    if output.exists():raise ValueError('保留旧探针，请使用新目录')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='7':raise ValueError('仅允许物理卡7')
    import torch
    import torch_npu
    import triton
    from triton.runtime.jit import JITFunction
    assert torch.npu.device_count()==1
    output.mkdir(parents=True)
    m=load(source)
    record=dict(schema='hgkda-execution-slice-v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        operator=operator,probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physical_card=7,torch=str(torch.__version__),triton=str(triton.__version__),
        jit_runtime_sha256=hashlib.sha256(Path(inspect.getfile(JITFunction)).read_bytes()).hexdigest(),
        timing_scope='预热后批量同步walltime/调用，非纯设备时间；相同样本交替测量',cases=[])
    for dtype_name in ['float16','float32']:
        for shape in shapes:
            row=dict(id=dtype_name+'-'+'x'.join(map(str,shape)),shape=shape,dtype=dtype_name,status='failed')
            record['cases'].append(row)
            try:
                dtype=getattr(torch,dtype_name)
                original=torch.randn(shape,generator=torch.Generator().manual_seed(626132)).to(dtype)
                expected=getattr(torch,operator[:-1])(original)
                x=original.to('npu:0')
                for _ in range(4):x.copy_(original);m.run(x)
                x.copy_(original)
                captured,_=capture_call(m.run,x)
                if not captured:raise ValueError('没有捕获到JIT调用；执行路径待独立调查')
                direct=direct_call(captured)
                x.copy_(original);direct();torch.npu.synchronize()
                torch.testing.assert_close(x.cpu(),expected,atol=.015 if dtype_name=='float16' else 1e-4,
                                           rtol=.015 if dtype_name=='float16' else 1e-4,equal_nan=True)
                timings=paired_measure(dict(wrapper_us=lambda:m.run(x),direct_us=direct),lambda:x.copy_(original))
                launches=[]
                for item in captured:
                    jit=item['jit']; compiled=item['compiled']
                    binary={k:hashlib.sha256(v if isinstance(v,bytes) else str(v).encode()).hexdigest()
                            for k,v in compiled.asm.items()}
                    launches.append(dict(symbol=jit.__name__,grid=list(item['grid']),
                        programs=math.prod(item['grid']),parameters=list(item['bound']),
                        constants={k:v for k,v in item['bound'].items() if isinstance(v,(int,float,str,bool))},
                        compiled_artifact_sha256=binary,source_file=inspect.getfile(jit.fn)))
                row.update(status='passed',launches=launches,timings=timings,direct_semantics='passed')
                # 只对清晰的一维指针+长度接口开展分块探针，未知生成接口不猜参数。
                item=captured[0]
                if x.numel()>=65536 and len(captured)==1 and 'BLOCK_SIZE' in item['bound'] and 'n_elements' in item['bound'] and len(item['grid'])==1:
                    tile=[]
                    for block in [512,2048,4096]:
                        if block==item['bound']['BLOCK_SIZE']:continue
                        kwargs=dict(item['kwargs']);kwargs['BLOCK_SIZE']=block
                        grid=(triton.cdiv(x.numel(),block),)
                        if grid[0]>65535:continue
                        def alternate(kwargs=kwargs,grid=grid):
                            return JITFunction.run(item['jit'],*item['args'],grid=grid,warmup=False,**kwargs)
                        def same_dispatch_parent():
                            return JITFunction.run(item['jit'],*item['args'],grid=item['grid'],warmup=False,**item['kwargs'])
                        try:
                            x.copy_(original);alternate();torch.npu.synchronize()
                            torch.testing.assert_close(x.cpu(),expected,atol=.015 if dtype_name=='float16' else 1e-4,
                                rtol=.015 if dtype_name=='float16' else 1e-4,equal_nan=True)
                            paired=paired_measure(dict(parent=same_dispatch_parent,variant=alternate),lambda:x.copy_(original),trials=5,batch=8)
                            tile.append(dict(block=block,grid=list(grid),timings=paired,status='passed',
                                scope='同输入同JIT分派路径，仅分块与grid改变；不等同完整候选'))
                        except Exception as e:tile.append(dict(block=block,status='failed',error=type(e).__name__+': '+str(e)))
                    row['tile_interventions']=tile
                if trace and len(record['cases'])==1:
                    from torch_npu.profiler import profile,ProfilerActivity,tensorboard_trace_handler
                    x.copy_(original);torch.npu.synchronize()
                    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],
                                 on_trace_ready=tensorboard_trace_handler(str(output/'profiler'))) as prof:
                        with torch.autograd.profiler.record_function('hgkda_candidate_region'):
                            m.run(x)
                        torch.npu.synchronize()
                    prof.export_chrome_trace(str(output/'candidate-trace.json'))
                    row['trace']='candidate-trace.json'
            except Exception as e:row['error']=type(e).__name__+': '+str(e)
            (output/'probe.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    diagnosis=diagnose(record)
    (output/'diagnosis.json').write_text(json.dumps(diagnosis,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=sum(x['status']=='passed' for x in record['cases']),total=len(record['cases'])),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--operator',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shapes',required=True)
    p.add_argument('--trace',action='store_true')
    a=p.parse_args();probe(a.source,a.operator,a.output,json.loads(a.shapes),a.trace)
