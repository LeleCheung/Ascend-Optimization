"""执行一个冻结输入的候选或CANN参考，保存候选实际编译产物。"""
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
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--side',choices=['candidate','reference'],required=True)
    args=parser.parse_args()
    spec=json.loads(args.spec.read_bytes())
    import torch,torch_npu
    torch.set_num_threads(1);torch.npu.set_device(0)
    oracle=imported('_skill_oracle',Path(spec['leaf'])/'oracle.py')
    candidate=imported('_skill_source',spec['source']) if args.side=='candidate' else None
    inputs=oracle.gen_inputs(spec['case'],'npu:0')
    if candidate is None:
        oracle.run(**inputs);torch.npu.synchronize();return
    from triton.runtime.jit import JITFunction
    original=JITFunction.run
    artifacts=[]

    def capture(kernel,*values,**options):
        compiled=original(kernel,*values,**options)
        if compiled is not None and hasattr(compiled,'asm'):
            folder=Path(spec['ir_directory']);folder.mkdir(parents=True,exist_ok=True)
            for kind,value in compiled.asm.items():
                if not isinstance(value,(str,bytes)):continue
                raw=value.encode() if isinstance(value,str) else value
                target=folder/(kernel.__name__+'.'+kind)
                if target.exists():assert target.read_bytes()==raw
                else:target.write_bytes(raw)
                if kind in ['ttir','bcmlir']:
                    artifacts.append(dict(path=target.name,sha256=hashlib.sha256(raw).hexdigest()))
        return compiled
    JITFunction.run=capture
    try:
        candidate.run(**inputs);torch.npu.synchronize()
    finally:JITFunction.run=original
    value=dict(source_sha256=spec['source_sha256'],files=artifacts)
    (Path(spec['ir_directory'])/'manifest.json').write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
