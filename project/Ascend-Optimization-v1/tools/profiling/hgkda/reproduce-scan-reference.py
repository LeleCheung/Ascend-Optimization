#!/usr/bin/env python3
"""固定字节复现累计积分参考不稳定，输出原容差无公共交集的逐点证据。"""
import argparse
import hashlib
import json
from pathlib import Path
import os
import sys
sys.dont_write_bytecode=True


def main():
    import numpy as np
    import torch
    import torch_npu
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--deterministic',action='store_true')
    a=p.parse_args()
    assert not a.output.exists()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7' and torch.npu.device_count()==1
    data=a.input.read_bytes()
    expected='f5874ea96d99dcc5b9399c8ccea5a50015d99d1e52936425b28d65e842b3ed19'
    assert hashlib.sha256(data).hexdigest()==expected
    y=torch.from_numpy(np.frombuffer(data,dtype=np.float16).copy().reshape(256,8)).to('npu:0')
    torch.use_deterministic_algorithms(a.deterministic)
    outputs=[]
    # 只测参考，不调用候选，也不设定候选专有状态。
    for _ in range(16):
        value=torch.cumulative_trapezoid(y,dx=1,dim=0)
        torch.npu.synchronize()
        outputs.append(value.cpu().double().numpy())
    stack=np.stack(outputs)
    radii=.015+.015*np.abs(stack)
    lower=(stack-radii).max(0)
    upper=(stack+radii).min(0)
    empty=lower>upper
    points=[]
    for position in np.argwhere(empty)[:16]:
        at=tuple(int(x) for x in position)
        points.append(dict(position=list(at),sixteen_reference_values=[float(x[at]) for x in stack],
            maximum_lower_bound=float(lower[at]),minimum_upper_bound=float(upper[at]),
            gap=float(lower[at]-upper[at])))
    result=dict(status='reference_inconsistent' if empty.any() else 'no_counterexample_in_this_run',
        input_sha256=expected,shape=[256,8],dtype='float16',dim=0,dx=1,repetitions=16,
        tolerance=dict(atol=.015,rtol=.015,required_matched_ratio=1.),
        deterministic=a.deterministic,bitwise_stable=all(np.array_equal(stack[0],x) for x in stack[1:]),
        empty_intersections=int(empty.sum()),counterexamples=points,
        maximum_repeat_difference=float((stack.max(0)-stack.min(0)).max()),physical_card=7,
        torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,
        conclusion='区间为空的位置不存在同时满足这些参考输出的固定结果；无反例不等于证明默认稳定')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(a.output.with_suffix('.outputs.npy'),stack)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
