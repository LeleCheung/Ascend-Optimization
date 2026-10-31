"""仅CPU核对固定求逆输入的LAPACK中间值，不生成NPU候选成绩。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch

def tensor(t):
    value=t.detach().cpu().contiguous()
    return dict(shape=list(value.shape),dtype=str(value.dtype),values=value.tolist(),
        bytes_sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();data=json.loads(a.probe.read_text());rows=[]
    torch.set_num_threads(1)
    for case in data['cases']:
        if not case.get('input',{}).get('values'):continue
        matrix=torch.tensor(case['input']['values'],dtype=torch.float32,device='cpu')
        for transposed in [False,True]:
            source=matrix.mT.contiguous() if transposed else matrix.contiguous()
            lu,piv,info=torch.linalg.lu_factor_ex(source,check_errors=False)
            identity=torch.eye(matrix.shape[-1],dtype=matrix.dtype).expand_as(matrix).clone()
            inverse=torch.linalg.lu_solve(lu,piv,identity,adjoint=transposed)
            reference,refinfo=torch.linalg.inv_ex(matrix,check_errors=False)
            rows.append(dict(case=case['case'],transposed=transposed,lu=tensor(lu),pivots=tensor(piv),info=tensor(info),
                solved=tensor(inverse),cpu_inverse=tensor(reference),cpu_info=tensor(refinfo),
                original_record_reference=case['outputs'][0]['npu_reference']['values']))
    result=dict(status='cpu_lapack_intermediates',rows=rows,torch_version=torch.__version__,torch_revision=torch.version.git_version,
        build_config=torch.__config__.show(),probe_sha256=hashlib.sha256(a.probe.read_bytes()).hexdigest(),
        script_source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        scope='单CPU线程、无NPU调用；用于核对有限精度和求解顺序，不作为优化候选或正式成绩。')
    def clean(v):
        if isinstance(v,float) and not math.isfinite(v):return str(v)
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        return v
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf8')
    print(json.dumps(dict(rows=len(rows),status=result['status'])))

if __name__=='__main__':main()
