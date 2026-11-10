"""log10专项：验证极小值、特殊值、数学边界与尾块，不修改正式合同或候选。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','leaf','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    import torch,torch_npu
    torch.set_num_threads(1);torch.npu.set_device(0)
    candidate=imported('_edge_candidate',args.source)
    oracle=imported('_edge_reference',args.leaf/'oracle.py')
    contract=[json.loads(line) for line in (args.leaf/'correctness.jsonl').read_text().splitlines() if line.strip()]
    rows=[]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        info=torch.finfo(dtype)
        values=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),-1.,1.,
            info.tiny,info.tiny*info.eps,info.max,info.eps,1-info.eps,1+info.eps,-info.tiny],dtype=dtype)
        boundaries=torch.tensor([2/3,4/3,1.,2.,info.tiny],dtype=dtype)
        values=torch.cat([values,boundaries,
            torch.nextafter(boundaries,torch.full_like(boundaries,float('inf'))),
            torch.nextafter(boundaries,torch.full_like(boundaries,-float('inf')))])
        tolerance=next(c['tolerance'] for c in contract if c['inputs']['spec']['dtype']==str(dtype).split('.')[-1])
        for strided in [False,True]:
            row=dict(dtype=str(dtype),strided=strided,elements=values.numel(),passed=False)
            try:
                backing=torch.full((2*values.numel()+3,),37.,dtype=dtype,device='npu:0')
                tensor=backing[1:1+2*values.numel():2] if strided else values.to('npu:0')
                if strided:tensor.copy_(values)
                expected=oracle.run(A=values.to('npu:0'))
                actual=candidate.run(A=tensor);torch.npu.synchronize()
                torch.testing.assert_close(actual,expected,atol=tolerance['atol'],rtol=tolerance['rtol'],equal_nan=True)
                assert torch._C._is_alias_of(actual,tensor),'原地返回别名错误'
                if strided:
                    assert torch.all(backing[::2]==37).item() and torch.all(backing[-3:]==37).item(),'写坏非目标存储'
                row['passed']=True
            except Exception as error:row['error']=type(error).__name__+': '+str(error)[:1800]
            rows.append(row)
        # 非整除大输入独立检查，覆盖单块／流式路径切换及不满的最后一块。
        for elements in [24577,65535,65537]:
            row=dict(dtype=str(dtype),strided=False,offset=1,elements=elements,check='large_partial_tile',passed=False)
            try:
                values=torch.linspace(.5,2.,elements,dtype=torch.float32).to(dtype)
                backing=torch.full((elements+3,),37.,dtype=dtype,device='npu:0')
                tensor=backing[1:elements+1];tensor.copy_(values)
                expected=oracle.run(A=values.to('npu:0'))
                actual=candidate.run(A=tensor);torch.npu.synchronize()
                torch.testing.assert_close(actual,expected,atol=tolerance['atol'],rtol=tolerance['rtol'],equal_nan=True)
                assert torch._C._is_alias_of(actual,tensor),'原地返回别名错误'
                assert backing[0].item()==37 and torch.all(backing[-2:]==37).item(),'写坏邻接存储'
                row['passed']=True
            except Exception as error:row['error']=type(error).__name__+': '+str(error)[:1800]
            rows.append(row)
    result=dict(status='passed' if all(row['passed'] for row in rows) else 'failed',
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference='固定合同中的NPU CANN参考',formal_contract_changed=False,
        passed=sum(row['passed'] for row in rows),total=len(rows),cases=rows)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:result[key] for key in ['status','passed','total','source_sha256']}))


if __name__=='__main__':main()
