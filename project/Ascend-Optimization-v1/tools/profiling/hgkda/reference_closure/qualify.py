"""按冻结输入逐项核验CANN参考，不把局部支持说成完整合同通过。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['operator','source','leaf','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output);source=Path(a.source);leaf=Path(a.leaf)
    if out.exists():raise FileExistsError(out)
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    import torch,torch_npu
    oracle=load('closure_oracle',leaf/'oracle.py');candidate=load('closure_reference',source)
    cases=[json.loads(x) for x in (leaf/'correctness.jsonl').read_text(encoding='utf8').splitlines() if x.strip()]
    # 所有正确性输入先核验；性能输入另在完整KGS中检验，不用短向量代替资格。
    rows=[]
    for case in cases:
        row=dict(case=case['name'],spec=case['inputs']['spec'],passed=False)
        try:
            args=oracle.gen_inputs(case,'npu:0');refargs=oracle.gen_inputs(case,'npu:0')
            before={k:v.clone() for k,v in args.items() if isinstance(v,torch.Tensor)}
            expected=oracle.run(**refargs);actual=candidate.run(**args);torch.npu.synchronize()
            expected=list(expected) if isinstance(expected,(tuple,list)) else [expected]
            actual=list(actual) if isinstance(actual,(tuple,list)) else [actual]
            row['outputs']=[]
            if a.operator=='linalg_inv_ex':
                ref=expected[0];value=actual[0]
                row['inverse_matched']=bool(torch.allclose(value,ref,atol=case['tolerance']['atol'],rtol=case['tolerance']['rtol'],equal_nan=True))
                product=args['A'].float()@value.float();eye=torch.eye(args['A'].shape[-1],device=value.device)
                row['residual_max']=float((product-eye).abs().max().cpu());row['reference_info']=expected[1].cpu().tolist()
                row['partial_only']=True;row['passed']=row['inverse_matched'];row['info_supported']=False
            else:
                assert len(expected)==len(actual),'输出数量不同'
                for i,(r,v) in enumerate(zip(expected,actual)):
                    item=dict(output=i,expected_dtype=str(r.dtype),actual_dtype=str(v.dtype),shape=list(v.shape))
                    row['outputs'].append(item);assert r.shape==v.shape and r.dtype==v.dtype
                    torch.testing.assert_close(v,r,atol=case['tolerance']['atol'],rtol=case['tolerance']['rtol'],equal_nan=True)
                    item['passed']=True
                if a.operator=='expm1_':assert actual[0] is args['A'],'原地返回对象错误'
                elif a.operator=='fake_quantize_per_tensor_affine_cachemask_out':
                    assert actual[0] is args['out0'] and actual[1] is args['out1'],'out别名错误'
                for k,v in before.items():
                    if k in {'out0','out1'} or a.operator=='expm1_':continue
                    torch.testing.assert_close(args[k],v,atol=0,rtol=0,equal_nan=True)
                row['passed']=True
        except Exception as e:row['error']=type(e).__name__+': '+str(e)[:2000]
        rows.append(row)
    full=a.operator!='linalg_inv_ex' and all(r['passed'] for r in rows)
    result=dict(status='qualified' if full else 'partial_reference' if a.operator=='linalg_inv_ex' else 'unsupported_or_mismatch',operator=a.operator,
        source=str(source),source_sha256=sha(source),contract_sha256={n:sha(leaf/n) for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']},
        eligible_full_contract=full,passed=sum(r['passed'] for r in rows),total=len(rows),cases=rows)
    def standard(v):
        if isinstance(v,float) and not __import__('math').isfinite(v):return str(v)
        if isinstance(v,list):return [standard(x) for x in v]
        if isinstance(v,dict):return {k:standard(x) for k,x in v.items()}
        return v
    result=standard(result)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))
if __name__=='__main__':main()
