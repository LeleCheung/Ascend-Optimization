"""Check original real-input alias behavior against both frozen references."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    candidate=load(a.source,'native_conj_alias_check')
    reference=load(a.reference,'cann_conj_alias_check')
    x=torch.arange(17,dtype=torch.float32,device='npu:0')
    result=candidate.run(x)
    cann=reference.run(x)
    pytorch=torch.ops.aten._conj_physical(x)
    torch.npu.synchronize()
    rows=dict(candidate_alias=torch._C._is_alias_of(result,x),
              cann_alias=torch._C._is_alias_of(cann,x),
              pytorch_alias=torch._C._is_alias_of(pytorch,x),
              values_match=bool(torch.equal(result.cpu(),cann.cpu()) and torch.equal(result.cpu(),pytorch.cpu())))
    passed=rows['candidate_alias']==rows['cann_alias']==rows['pytorch_alias'] and rows['values_match']
    a.output.write_text(json.dumps(dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        status='passed' if passed else 'failed',passed=int(passed),total=1,case='real_input_output_alias',
        observed=rows,scope='Additional real-input storage semantics; does not replace historical value-only measurements'),indent=2)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':main()
