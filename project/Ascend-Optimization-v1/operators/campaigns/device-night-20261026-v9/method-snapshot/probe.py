"""同输入、同计时入口的设备分块干预；保存编译IR、设备延迟与失败。"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'mechanism'))
from variants import transform
from measure import pair


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def compiled_evidence(records,out):
    saved=[]
    for i,item in enumerate(records):
        compiled=item['compiled'];directory=out/('kernel-'+str(i));directory.mkdir(parents=True,exist_ok=True)
        artifacts=[]
        for name,data in compiled.asm.items():
            if not str(name).replace('_','').isalnum():continue
            payload=data if isinstance(data,bytes) else str(data).encode()
            path=directory/(name+('.bin' if isinstance(data,bytes) else '.txt'));path.write_bytes(payload)
            artifacts.append(dict(path=str(path),kind=name,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
        saved.append(dict(symbol=item['jit'].fn.__name__,grid=list(item['grid']),artifacts=artifacts,
            constants={k:v for k,v in item['bound'].items() if type(v) in (int,float,str,bool)},
            metadata=str(compiled.metadata)))
    return saved


def collect(op,source,config_path,case_path,out):
    if out.exists():raise ValueError('请用新探针目录')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='7':raise ValueError('仅允许物理卡7')
    import torch
    import torch_npu
    from triton.backends.ascend.testing import do_bench_npu
    # 仅诊断进程临时截获JIT，候选源码不修改全局类。
    parent_probe=load(HERE.parent/'mechanism/probe.py','device_capture_helpers') if (HERE.parent/'mechanism/probe.py').exists() else load(HERE/'launch-probe.py','device_capture_helpers')
    oracle=load(case_path.parent/'oracle.py','device_oracle_'+op)
    configs=json.loads(config_path.read_text());cases=json.loads(case_path.read_text())
    out.mkdir(parents=True)
    baseline=load(source,'device_parent_'+op)
    data=dict(schema='hgkda-device-interventions-v1',operator=op,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        physical_card=7,scope='固定输入指针、捕获的Triton提交；do_bench_npu设备kernel计时，非完整调用walltime',
        measurement_backend_sha256=hashlib.sha256(inspect.getsource(do_bench_npu).encode()).hexdigest(),
        torch=str(torch.__version__),cases=[],started_at=time.time())
    for ci,case in enumerate(cases):
        ctx=dict(inputs={'spec':case},seed=71439+ci)
        inputs=oracle.gen_inputs(ctx,'npu:0');reference=oracle.run(**inputs);torch.npu.synchronize()
        calls,base_result=parent_probe.capture_call(lambda _:baseline.run(**inputs),None)
        if len(calls)!=1:raise ValueError('父路径不是一个可捕获JIT')
        tolerance=1e-4*case['shape'][-1] if op=='matmul_bias_activation' else 0
        rtol={'float16':1e-3,'bfloat16':.016,'float32':1.3e-6}[case['dtype']] if tolerance else 0
        torch.testing.assert_close(base_result,reference,atol=tolerance,rtol=rtol,equal_nan=True)
        base_direct=parent_probe.direct_call(calls)
        entry=dict(id=str(ci),spec=case,parent_launches=compiled_evidence(calls,out/('case-'+str(ci))/'parent'),variants=[])
        data['cases'].append(entry);save(out/'probe.json',data)
        for vi,choice in enumerate(configs):
            row=dict(name=choice['name'],config=choice['config'],status='failed');entry['variants'].append(row)
            directory=out/('case-'+str(ci))/choice['name'];directory.mkdir(parents=True)
            candidate=directory/'candidate.py';candidate.write_bytes(transform(source.read_text(encoding='utf8'),op,choice['config']).encode())
            row['source_sha256']=hashlib.sha256(candidate.read_bytes()).hexdigest()
            try:
                module=load(candidate,'device_variant_'+op+'_'+str(ci)+'_'+str(vi))
                captured,actual=parent_probe.capture_call(lambda _:module.run(**inputs),None)
                if len(captured)!=1:raise ValueError('候选路径不是一个可捕获JIT')
                torch.testing.assert_close(actual,reference,atol=tolerance,rtol=rtol,equal_nan=True)
                # 保持编译对象与输入固定，只改变被登记的设备组织。
                variant_direct=parent_probe.direct_call(captured)
                row['launches']=compiled_evidence(captured,directory)
                timings=dict(parent_ms=[],variant_ms=[])
                for repeat in range(3):
                    funcs=[('parent_ms',base_direct),('variant_ms',variant_direct)]
                    if repeat%2:funcs.reverse()
                    measured=pair(dict(funcs),dict(parent_ms=calls[0]['jit'].fn.__name__,variant_ms=captured[0]['jit'].fn.__name__),directory/('benchmark-'+str(repeat)))
                    for key,number in measured.items():timings[key].append(number)
                row.update(status='passed',timings=timings,paired_speedup=statistics.median([a/b for a,b in zip(timings['parent_ms'],timings['variant_ms'])]),
                    max_abs_error=float((actual.float()-reference.float()).abs().max().item()))
            except Exception as error:
                row['error']=type(error).__name__+': '+str(error)[-3000:]
                # 异步设备错误不能作为编译失败跳过。
                torch.npu.synchronize()
            save(out/'probe.json',data)
            print(json.dumps(dict(case=ci,variant=choice['name'],status=row['status'],speedup=row.get('paired_speedup'),error=row.get('error')),ensure_ascii=False),flush=True)
        del inputs,reference,base_result,calls
    data['finished_at']=time.time();save(out/'probe.json',data)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--operator',required=True)
    for name in ['source','configs','cases','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();collect(a.operator,a.source,a.configs,a.cases,a.output)
