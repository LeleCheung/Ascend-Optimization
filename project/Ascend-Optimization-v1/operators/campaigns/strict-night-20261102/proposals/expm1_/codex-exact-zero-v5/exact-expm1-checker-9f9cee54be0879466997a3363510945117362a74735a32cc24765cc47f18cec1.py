"""补查8192整块实际路径的小值、负零与布局语义，不做性能计时。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

DTYPES=('float16','bfloat16','float32')
LAYOUTS=('exact8192','exact16384','offset8192','tail8193')


def expected_cases():
    return {dtype+'-'+layout for dtype in DTYPES for layout in LAYOUTS}


def qualified(value,digest):
    rows=value.get('cases',[])
    return (value.get('status')=='passed' and value.get('source_sha256')==digest
        and value.get('passed')==value.get('total')==12 and len(rows)==12
        and {row.get('case') for row in rows}==expected_cases()
        and all(row.get('passed') is True and row.get('zero_sign_checked') is True for row in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    assert not args.output.exists(), '已有补测身份，不重发'
    args.output.mkdir(parents=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    identity=dict(source=str(args.source),source_sha256=sha(args.source),checker_sha256=sha(__file__))
    (args.output/'started.json').write_text(json.dumps(identity)+'\n',encoding='utf8')
    import torch
    import torch_npu
    spec=importlib.util.spec_from_file_location('exact_expm1_candidate',args.source)
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    values=[0.,-0.,1e-8,-1e-8,1e-6,-1e-6,1e-4,-1e-4,.006,-.006,.0099,-.0099,
            .01,-.01,.0101,-.0101,.03,-.03,.0624,-.0624,.0625,-.0625,float('nan'),float('inf'),-float('inf')]
    rows=[]
    for dtype_name in DTYPES:
        dtype=getattr(torch,dtype_name)
        for layout in LAYOUTS:
            n=16384 if layout=='exact16384' else 8193 if layout=='tail8193' else 8192
            offset=1 if layout=='offset8192' else 0
            row=dict(case=dtype_name+'-'+layout,elements=n,offset=offset)
            try:
                cpu=torch.tensor(values,dtype=dtype).repeat((n+len(values)-1)//len(values))[:n]
                reference=torch.expm1(cpu)
                backing=torch.full((n+offset+16,),42.,dtype=dtype,device='npu:0')
                x=backing[offset:offset+n]
                x.copy_(cpu)
                before=backing.cpu()
                returned=candidate.run(x)
                torch.npu.synchronize()
                assert returned is x, '原地返回对象变化'
                actual=x.cpu()
                rtol=1e-4 if dtype==torch.float32 else .015
                torch.testing.assert_close(actual,reference,rtol=rtol,atol=0,equal_nan=True)
                zeros=cpu==0
                assert torch.equal(torch.signbit(actual[zeros]),torch.signbit(reference[zeros])), '零符号变化'
                after=backing.cpu()
                assert torch.equal(after[:offset],before[:offset])
                assert torch.equal(after[offset+n:],before[offset+n:]), '写出了目标view'
                row.update(passed=True,rtol=rtol,atol=0,zero_sign_checked=True,
                           input_sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())
            except Exception as error:
                row.update(passed=False,error=type(error).__name__+': '+str(error))
            rows.append(row)
    result=dict(identity,status='passed' if all(row['passed'] for row in rows) else 'failed',
                passed=sum(row['passed'] for row in rows),total=len(rows),cases=rows,
                scope='8192/16384实际完整块及非对齐/尾块，独立CPU小值和零符号参考；补充正确性检查，不计时、不改变原合同或CANN资格。')
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],passed=result['passed'],total=result['total'])),flush=True)


if __name__=='__main__':main()
