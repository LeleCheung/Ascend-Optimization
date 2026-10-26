#!/usr/bin/env python3
"""验证本批两项NPU分解对照与交付源码，保留默认参考和所有原合同字节。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True

ROOT=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
BATCH=ROOT/'operators/campaigns/acceptance-three-20261026'
TOOLS=ROOT/'tools/profiling/hgkda'


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+chr(10))


def assert_equal(first,second):
    import torch
    if isinstance(first,torch.Tensor):
        torch.testing.assert_close(first.cpu(),second.cpu(),atol=0,rtol=0,equal_nan=True)
        assert first.device.type=='npu'
    elif isinstance(first,(tuple,list)):
        assert type(first)==type(second) and len(first)==len(second)
        for a,b in zip(first,second):assert_equal(a,b)
    else:assert first==second


def run(op):
    import torch
    import torch_npu
    from kernelgen_client.catalog import Catalog
    from kernelgen_client.protocol.schema import Implementation,EvaluationSettings
    from kernelgen_server.evaluation.engine import EvaluationEngine
    from kernelgen_server.backends.ascend.device import NpuDevice
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7' and torch.npu.device_count()==1
    assert not torch.are_deterministic_algorithms_enabled()
    out=BATCH/'validation-v2'/op
    out.mkdir(parents=True,exist_ok=False)
    kind='pool-variants' if op=='fractional_max_pool3d' else 'conj-variants'
    selection=json.loads((BATCH/'research'/kind/'selection.json').read_text())
    original=Path(selection['best']['source'])
    delivered=BATCH/'deliverables'/op/'main.py'
    delivered.parent.mkdir(parents=True,exist_ok=True)
    if delivered.exists():assert delivered.read_bytes()==original.read_bytes()
    delivered.write_bytes(original.read_bytes())
    assert sha(delivered)==selection['best']['source_sha256']
    source=delivered
    gates=load('acceptance_gate',TOOLS/'check-remaining-five.py')
    assert gates.check(op,source,out/'semantics.json')==0
    if op=='fractional_max_pool3d':
        special=load('acceptance_special',TOOLS/'check-remaining-special.py')
        assert special.check(op,source,out/'special.json')==0
    # 第二套合同仅替换oracle入口为明确命名的自建NPU分解，原目录始终不改。
    leaf=BATCH/'catalog/ops'/op
    target=BATCH/'npu-decomposition-catalog-v2/ops'/op
    target.mkdir(parents=True,exist_ok=False)
    for path in leaf.iterdir():
        if path.is_file():shutil.copyfile(path,target/path.name)
    ref_path=TOOLS/'npu-reference-decompositions-v2.py'
    helper=ref_path.read_text()
    oracle=(leaf/'oracle.py').read_text()
    fn='fractional_pool' if op=='fractional_max_pool3d' else 'conjugate'
    # 参考文件完整自包含；保留原gen_inputs，每次新数据，不缓存输出。
    oracle+=chr(10)+helper+chr(10)+'run='+fn+chr(10)
    (target/'oracle.py').write_text(oracle)
    for path in leaf.iterdir():
        if path.name!='oracle.py' and path.is_file():assert path.read_bytes()==(target/path.name).read_bytes()
    save(out/'conditions.json',dict(operator=op,source_sha256=sha(source),parent_source=str(original),
        reference_scope='补充自建纯NPU PyTorch原语分解；非官方单算子；不替代默认参考验收',
        reference_sha256=sha(target/'oracle.py'),decomposition_sha256=sha(ref_path),
        original_contract_sha256={p.name:sha(p) for p in leaf.iterdir() if p.is_file()},
        supplemental_contract_sha256={p.name:sha(p) for p in target.iterdir() if p.is_file()},
        physical_device=7,timing_scope='原KGS Native引擎完整调用walltime',deterministic=False))
    # 同一候选必须同时对默认参考与补充NPU参考数值成立；全部性能输入也验数值。
    candidate=load('acceptance_delivered',source)
    ref=load('acceptance_decomposed_oracle',target/'oracle.py')
    baseline=load('acceptance_original_oracle',leaf/'oracle.py')
    stream=torch.npu.Stream()
    checks=[]
    for phase in ['correctness','timing']:
        for case in [json.loads(l) for l in (leaf/(phase+'.jsonl')).read_text().splitlines()]:
            for repeat in range(2):
                args=baseline.gen_inputs(case,'npu:0')
                expected=baseline.run(**args)
                with torch.npu.stream(stream if repeat else torch.npu.default_stream()):
                    decomposed=ref.run(**args)
                    actual=candidate.run(**args)
                torch.npu.synchronize()
                assert_equal(decomposed,expected)
                assert_equal(actual,expected)
            checks.append(dict(case=case['name'],phase=phase,status='passed',repetitions=2,stream_switch=True))
    save(out/'paired-numerics.json',dict(status='passed',total=len(checks),passed=len(checks),cases=checks,source_sha256=sha(source)))
    def evaluate(catalog_root,label):
        catalog=Catalog(catalog_root)
        definition=catalog.load(op)
        implementation=Implementation(name='acceptance-'+label,definition=op,language='triton',
            entrypoint='main.py::run',sources=[dict(path='main.py',content=source.read_text())])
        request=catalog.request(op,implementation,EvaluationSettings(warmup_ms=1000,benchmark_ms=100,num_trials=3,timeout_seconds=1200))
        save(out/(label+'.request.json'),request.model_dump(mode='json'))
        engine=EvaluationEngine(NpuDevice(perf_mode='walltime',num_trials=3),'npu:0','ascend',oracle_path=definition.oracle_path)
        preflight=engine.preflight(request)
        save(out/(label+'.preflight.json'),preflight)
        assert preflight['status']=='PASSED',preflight
        result=engine.evaluate(request).model_dump(mode='json')
        save(out/(label+'.result.json'),result)
        assert result['status']=='PASSED' and result['num_passed']==result['num_workloads'] and not result['is_hack'],result['status']
        print(label,json.dumps({k:result[k] for k in ['status','geo_mean','num_passed','num_workloads']}),flush=True)
        return result
    # 两种参考一份源码各一次正式完整评测，分别记分，绝不合并高分。
    original_result=evaluate(BATCH/'catalog','default-reference')
    npu_result=evaluate(BATCH/'npu-decomposition-catalog-v2','npu-decomposition-reference')
    observe=load('acceptance_observe',TOOLS/'observe-endurance.py')
    index=1 if op=='fractional_max_pool3d' else 3
    observe.observe(op,source,target,out/'npu-paired-path',index)
    save(out/'summary.json',dict(status='passed',operator=op,source_sha256=sha(source),
        default_reference_score=original_result['geo_mean'],npu_decomposition_score=npu_result['geo_mean'],
        num_passed=npu_result['num_passed'],reference_policy='两组分开；默认CPU回退仍不计正常NPU验收',
        adopted_method=selection['best']['label']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator',choices=['fractional_max_pool3d','_conj_physical'],required=True)
    run(p.parse_args().operator)
