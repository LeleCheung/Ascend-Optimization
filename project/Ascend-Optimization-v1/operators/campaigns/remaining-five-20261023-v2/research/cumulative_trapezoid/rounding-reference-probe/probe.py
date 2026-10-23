"""在KGS串行槽内核对真实9项输入的NPU扫描与CPU舍入模型；非候选评测。"""
from pathlib import Path
import hashlib,importlib.util,json,os
import torch
import torch_npu

assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
assert torch.npu.device_count()==1
torch.set_num_threads(2)
batch=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/remaining-five-20261023-v2')
record=next(r for r in json.loads((batch/'plan.json').read_text())['records'] if r['operator']=='cumulative_trapezoid')
leaf=batch/record['catalog_leaf']
for name,digest in record['contract_sha256'].items():
    assert hashlib.sha256((leaf/name).read_bytes()).hexdigest()==digest
spec=importlib.util.spec_from_file_location('rounding_oracle',leaf/'oracle.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)


def stats(actual,expected,tol):
    a=actual.float();b=expected.float()
    close=torch.isclose(a,b,atol=tol['atol'],rtol=tol['rtol'],equal_nan=True)
    return dict(passed=bool(close.all()),mismatched=int((~close).sum()),
                total=close.numel(),max_abs=float((a-b).abs().max()),
                exactly_equal=int((a==b).sum()))


def sequential(inc,dtype):
    values=inc.to(dtype);acc=torch.zeros(values.shape[0],dtype=dtype)
    out=torch.empty_like(values)
    for i in range(values.shape[1]):
        acc=acc+values[:,i];out[:,i]=acc
    return out.to(inc.dtype)


rows=[]
for line in (leaf/'timing.jsonl').read_text().splitlines():
    case=json.loads(line);args=oracle.gen_inputs(case,'npu:0')
    y=args['y'];axis=args['dim']%y.ndim;length=y.shape[axis]
    expected=oracle.run(**args)
    inc=(y.narrow(axis,0,length-1)+y.narrow(axis,1,length-1))*(args['dx']/2)
    original_dtype_scan=inc.cumsum(axis)
    promoted_scan=inc.float().cumsum(axis).to(y.dtype)
    torch.npu.synchronize()
    want=expected.cpu().movedim(axis,-1).contiguous().reshape(-1,length-1)
    cpu=inc.cpu().movedim(axis,-1).contiguous().reshape(-1,length-1)
    tol=case['tolerance']
    models={
        'npu_original_dtype_cumsum':original_dtype_scan.cpu().movedim(axis,-1).contiguous().reshape_as(want),
        'npu_fp32_cumsum_then_cast':promoted_scan.cpu().movedim(axis,-1).contiguous().reshape_as(want),
        'cpu_sequential_input_dtype':sequential(cpu,cpu.dtype),
        'cpu_sequential_fp32':sequential(cpu,torch.float32),
        'cpu_fp64_prefix_then_cast':cpu.double().cumsum(-1).to(cpu.dtype),
        'cpu_input_dtype_cumsum':cpu.cumsum(-1),
    }
    row=dict(case=case['name'],spec=case['inputs']['spec'],
             models={name:stats(value,want,tol) for name,value in models.items()})
    rows.append(row);print(json.dumps(row),flush=True)
result=dict(scope='Codex追加的真实NPU参考与CPU舍入模型诊断；不计候选性能',
            physical_device=7,contract_sha256=record['contract_sha256'],
            probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=rows)
Path('rounding-models.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
