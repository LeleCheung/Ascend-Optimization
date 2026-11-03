"""核验新原生随机入口的原始区间语义；不修改边界、裁剪或计性能。"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
OUT=PROJECT/'operators/campaigns/strict-five-20261103/reference-probes/uniform-tensor-v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7';assert not (OUT/'numerics.json').exists()
    import torch
    import torch_npu
    value=json.loads((OUT/'build.json').read_text())
    assert sha(value['binary'])==value['binary_sha256'] and sha(value['source'])==value['source_sha256']
    spec=importlib.util.spec_from_file_location(value['module_name'],value['binary'])
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    rows=[]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        for low,high in [(-.01,.02),(.1,.9)]:
            for length in [1,2]:
                row=dict(dtype=str(dtype),from_=low,to=high,offset_tensor_length=length,passed=False)
                try:
                    seed=torch.tensor([1732],dtype=torch.uint64).to('npu:0')
                    offset=torch.tensor([0]*length,dtype=torch.uint64).to('npu:0')
                    output=torch.empty(131072,dtype=dtype,device='npu:0')
                    actual=native.uniform_tensor(output,low,high,seed,offset,0);torch.npu.synchronize()
                    assert actual is output
                    cpu=actual.cpu().to(torch.float32)
                    row.update(min=float(cpu.min()),max=float(cpu.max()),
                        outside_lower=int((cpu<low).sum()),outside_upper=int((cpu>=high).sum()))
                    assert row['outside_lower']==row['outside_upper']==0,'原生输出超出原始请求区间'
                    row['passed']=True
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2500]
                rows.append(row)
    result=dict(status='probe_completed',passed=sum(r['passed'] for r in rows),total=len(rows),cases=rows,
        checker_sha256=sha(__file__),build_sha256=sha(OUT/'build.json'),qualified_CANN=False,performance_measured=False,
        scope='只检查未曾探测的Tensor seed/offset原生入口；不改from/to，不裁剪输出。即使局部通过，仍需完整合同、统计/状态/布局与路径核验。')
    (OUT/'numerics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],passed=result['passed'],total=result['total'])))


if __name__=='__main__':main()
