"""Collect one fixed GQA invocation, preserving source and compiled IR identity."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--case',type=int,required=True)
    p.add_argument('--side',choices=['candidate','reference'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch,torch_npu
    torch.set_num_threads(1);torch.npu.set_device(0)
    c=json.loads(a.contract.read_bytes());case=next(x for x in c['cases'] if x['case_id']==a.case)
    gen=torch.Generator(device='cpu').manual_seed(1729+a.case)
    dtype=getattr(torch,case['dtype'][0])
    inputs=[(torch.rand(s,generator=gen)*2-1).to(dtype).to('npu:0') for s in case['input_shape']]
    a.output.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    if a.side=='reference':
        q,k,v=inputs;scale=case['attrs']['scaleValue']
        if scale<=0:scale=q.shape[-1]**-.5
        mask=None
        if case['attrs']['is_causal']:
            sq,sk=q.shape[1],k.shape[1]
            mask=(torch.arange(sk)[None,:]>torch.arange(sq)[:,None]+sk-sq).to('npu:0')
        torch.ops.npu.npu_fused_infer_attention_score(q,k,v,num_heads=q.shape[2],num_key_value_heads=k.shape[2],input_layout='BSND',scale=scale,atten_mask=mask,sparse_mode=0)
        torch.npu.synchronize()
    else:
        spec=importlib.util.spec_from_file_location('gqa_candidate',a.source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from triton.runtime.jit import JITFunction
        original=JITFunction.run
        def capture(kernel,*values,**options):
            compiled=original(kernel,*values,**options)
            if compiled is not None and hasattr(compiled,'asm'):
                for kind,value in compiled.asm.items():
                    if not isinstance(value,(str,bytes)):continue
                    raw=value.encode() if isinstance(value,str) else value
                    target=a.output/(kernel.__name__+'.'+kind)
                    if target.exists():assert target.read_bytes()==raw
                    else:target.write_bytes(raw)
                    artifacts.append({'path':target.name,'sha256':hashlib.sha256(raw).hexdigest()})
            return compiled
        JITFunction.run=capture
        try:module.run(*inputs,**case['attrs']);torch.npu.synchronize()
        finally:JITFunction.run=original
    (a.output/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        contract_sha256=hashlib.sha256(a.contract.read_bytes()).hexdigest(),case=case,side=a.side,
        completion='invocation and synchronize returned',files=artifacts),indent=2))


if __name__=='__main__':main()
