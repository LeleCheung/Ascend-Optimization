#!/usr/bin/env python3
"""共轭实数短向量DMA与既有Triton路径同输入诊断，选有效新源码。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
sys.dont_write_bytecode=True

ROOT=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
BATCH=ROOT/'operators/campaigns/acceptance-three-20261026'
TOOLS=ROOT/'tools/profiling/hgkda'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    import torch
    import torch_npu
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7' and torch.npu.device_count()==1
    out=BATCH/'research/conj-variants'
    out.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'operators/campaigns/remaining-five-20261023-v2'
    source=parent/'deliverables/reviewed/_conj_physical/main.py'
    original=source.read_text()
    candidates=[('parent',source)]
    for threshold in [4096,262144,2097152]:
        path=out/('dma-'+str(threshold)+'.py')
        prefix=original.replace('def run(input):','def _triton_run(input):',1)
        suffix='''
import ctypes as _ct
_rt = _ct.CDLL('libacl_rt.so')
_rt.aclrtMemcpyAsync.argtypes=[_ct.c_void_p,_ct.c_size_t,_ct.c_void_p,_ct.c_size_t,_ct.c_int,_ct.c_void_p]
_rt.aclrtMemcpyAsync.restype=_ct.c_int

def run(input):
    if input.numel() and input.is_contiguous() and (not input.is_complex() or input.is_conj()) and input.numel()<=DMA_THRESHOLD:
        src=input.conj() if input.is_conj() else input
        output=torch.empty_like(src)
        size=input.numel()*input.element_size()
        stream=torch.npu.current_stream(input.device)
        status=_rt.aclrtMemcpyAsync(_ct.c_void_p(output.data_ptr()),size,_ct.c_void_p(src.data_ptr()),size,3,_ct.c_void_p(stream.npu_stream))
        if status:raise RuntimeError('aclrtMemcpyAsync: '+str(status))
        input.record_stream(stream)
        output.record_stream(stream)
        return output
    return _triton_run(input)
'''.replace('DMA_THRESHOLD',str(threshold))
        path.write_text(prefix+suffix)
        candidates.append(('dma-'+str(threshold),path))
    common=load('conj_common',TOOLS/'probe-acceptance-three.py')
    reference=load('conj_decomp',TOOLS/'npu-reference-decompositions.py')
    oracle=load('conj_oracle',parent/'catalog/ops/_conj_physical/oracle.py')
    cases=[json.loads(l) for l in (parent/'catalog/ops/_conj_physical/timing.jsonl').read_text().splitlines()]
    rows=[]
    for label,path in candidates:
        module=load('conj_variant_'+label.replace('-','_'),path)
        row=dict(label=label,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cases=[],status='passed')
        try:
            for case in cases:
                args=oracle.gen_inputs(case,'npu:0')
                expected=reference.conjugate(**args)
                actual=module.run(**args)
                torch.npu.synchronize()
                torch.testing.assert_close(actual.cpu(),expected.cpu(),atol=0,rtol=0)
                measured=common.latency(lambda:module.run(**args),21)
                row['cases'].append(dict(case=case['name'],**measured))
        except Exception as error:
            row.update(status='failed',error=type(error).__name__+': '+str(error)[-2500:])
        rows.append(row)
        common.save(out/'results.json',dict(scope='同输入21次诊断选版；不作为KGS正式成绩',rows=rows))
        print(json.dumps(row),flush=True)
    import math
    valid=[r for r in rows if r['status']=='passed' and len(r['cases'])==len(cases)]
    for row in valid:
        row['geometric_latency_ms']=math.exp(sum(math.log(x['median_ms']) for x in row['cases'])/len(cases))
    best=min(valid,key=lambda r:r['geometric_latency_ms'])
    common.save(out/'selection.json',dict(best=best,all_valid=valid))


if __name__=='__main__':
    main()
