"""同输入重复参考与显式表达式对照；不改确定性设置、参考或容差。"""
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
for name,digest in record['contract_sha256'].items():assert hashlib.sha256((leaf/name).read_bytes()).hexdigest()==digest
spec=importlib.util.spec_from_file_location('stability_oracle',leaf/'oracle.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)


def compare(actual,expected,tol):
    a=actual.float();b=expected.float();close=torch.isclose(a,b,atol=tol['atol'],rtol=tol['rtol'],equal_nan=True)
    return dict(passed=bool(close.all()),mismatched=int((~close).sum()),
                max_abs=float((a-b).abs().max()),exactly_equal=int((a==b).sum()),total=a.numel())


def metadata(x):
    data=dict(shape=list(x.shape),strides=list(x.stride()),dtype=str(x.dtype))
    try:data['npu_format']=torch_npu.get_npu_format(x)
    except Exception as error:data['format_query_error']=str(error)
    return data


rows=[]
for line in (leaf/'timing.jsonl').read_text().splitlines():
    case=json.loads(line)
    if case['name'] not in ['cumulative_trapezoid-timing-1','cumulative_trapezoid-timing-2',
                            'cumulative_trapezoid-timing-5','cumulative_trapezoid-timing-8']:continue
    args=oracle.gen_inputs(case,'npu:0');y=args['y'];axis=args['dim']%y.ndim
    tol=case['tolerance'];outputs=[]
    for repetition in range(6):
        value=oracle.run(**args);torch.npu.synchronize();outputs.append(value.cpu())
    intervals_low=[];intervals_high=[]
    for value in outputs:
        v=value.double();error=tol['atol']+tol['rtol']*v.abs()
        intervals_low.append(v-error);intervals_high.append(v+error)
    low=torch.stack(intervals_low).max(0).values
    high=torch.stack(intervals_high).min(0).values
    impossible=low>high
    row=dict(case=case['name'],spec=case['inputs']['spec'],repeated_reference=[compare(v,outputs[0],tol) for v in outputs[1:]],
             empty_tolerance_intersections=int(impossible.sum()),max_interval_gap=float((low-high).max()))
    if bool(impossible.any()):
        index=tuple(int(i) for i in impossible.nonzero()[0])
        row['first_disjoint']=dict(index=index,reference_values=[float(v[index]) for v in outputs],
                                   interval_low=float(low[index]),interval_high=float(high[index]))
    left=y.narrow(axis,0,y.shape[axis]-1);right=y.narrow(axis,1,y.shape[axis]-1)
    sum_tensor=left+right
    inc_left=(args['dx']/2)*sum_tensor
    inc_right=sum_tensor*(args['dx']/2)
    torch.npu.synchronize()
    row['increments_equal']=torch.equal(inc_left.cpu(),inc_right.cpu())
    row['metadata']={'input':metadata(y),'sum':metadata(sum_tensor),'left_mul':metadata(inc_left),'right_mul':metadata(inc_right)}
    for label,inc in [('left_scalar',inc_left),('right_scalar',inc_right)]:
        repeated=[]
        for repetition in range(3):
            value=inc.cumsum(axis);torch.npu.synchronize();repeated.append(value.cpu())
        row[label]=dict(vs_reference=[compare(v,outputs[0],tol) for v in repeated],
                        self_repeat=[compare(v,repeated[0],tol) for v in repeated[1:]])
    rows.append(row);print(json.dumps(row),flush=True)
result=dict(scope='Codex同输入参考自洽检查；默认设置，不替换参考或容差，不计候选性能',
            physical_device=7,probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            contract_sha256=record['contract_sha256'],cases=rows)
Path('stability-models.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
