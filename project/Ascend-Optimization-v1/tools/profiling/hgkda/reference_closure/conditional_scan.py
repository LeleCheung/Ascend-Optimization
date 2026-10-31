"""隔离进程中对双方启用官方确定性条件，使用原KGS引擎和冻结合同。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')

def main():
    p=argparse.ArgumentParser()
    for name in ['source','leaf','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    import torch,torch_npu
    from kernelgen_client.catalog import Catalog
    from kernelgen_client.protocol.schema import Implementation,EvaluationSettings
    from kernelgen_server.evaluation.engine import EvaluationEngine
    from kernelgen_server.backends.ascend.device import NpuDevice
    a.output.mkdir(parents=True,exist_ok=True);op='cumulative_trapezoid';request_path=a.output/'deterministic-1.request.json';result_path=a.output/'deterministic-1.result.json'
    if request_path.exists():
        if not result_path.exists():raise RuntimeError('确定性评测请求结果未知，不重发')
        assert json.loads(request_path.read_text())['implementation']['sources'][0]['content']==a.source.read_text()
        return
    torch.use_deterministic_algorithms(True)
    catalog=Catalog(a.leaf.parents[1]);definition=catalog.load(op)
    impl=Implementation(name='HG-KDA-tiled-restore-v2-deterministic',definition=op,language='triton',entrypoint='main.py::run',sources=[dict(path='main.py',content=a.source.read_text())])
    request=catalog.request(op,impl,EvaluationSettings(warmup_ms=1000,benchmark_ms=100,num_trials=3,timeout_seconds=1800))
    save(a.output/'conditions.json',dict(source=str(a.source),source_sha256=sha(a.source),physical_device=7,deterministic=True,applies_to='reference_and_candidate',
        contract_sha256={n:sha(a.leaf/n) for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']},
        script_sha256=sha(__file__),timing_scope='KGS Native引擎完整调用walltime',scope='独立进程官方确定性条件；不替代默认合同交付'))
    save(request_path,request.model_dump(mode='json'))
    engine=EvaluationEngine(NpuDevice(perf_mode='walltime',num_trials=3),'npu:0','ascend',oracle_path=definition.oracle_path)
    preflight=engine.preflight(request);save(a.output/'preflight.json',preflight)
    if preflight['status']!='PASSED':raise RuntimeError('确定性准入失败')
    result=engine.evaluate(request).model_dump(mode='json');save(result_path,result)
    print(json.dumps({k:result.get(k) for k in ['status','geo_mean','num_passed','num_workloads']},ensure_ascii=False),flush=True)
    # 同一隔离进程中保留确定性条件，并用当前补测器检查新数据与stream。
    sys.argv=[str(HERE.parent/'next_phase/validate.py'),'--operator',op,'--source',str(a.source),'--leaf',str(a.leaf),'--output',str(a.output/'extra-checks.json')]
    runpy.run_path(str(HERE.parent/'next_phase/validate.py'),run_name='__main__')

if __name__=='__main__':main()
