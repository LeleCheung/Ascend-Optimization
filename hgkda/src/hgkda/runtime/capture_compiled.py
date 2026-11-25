"""Capture Triton artifacts from actual candidate calls and check that ATen computation was not invoked."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def imported(name, path):
    spec=importlib.util.spec_from_file_location(name, path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','leaf','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu
    from triton.runtime.jit import JITFunction
    from torch.utils._python_dispatch import TorchDispatchMode
    torch.set_num_threads(1)

    class CandidateOnly(TorchDispatchMode):
        def __init__(self):
            super().__init__(); self.operations=[]

        def __torch_dispatch__(self,function,types,args=(),kwargs=None):
            name=str(function); self.operations.append(name)
            allowed={'aten.empty','aten.empty_like','aten.empty_strided','aten.alias','aten.detach',
                     'aten.view','aten._unsafe_view','aten.as_strided','aten.slice','aten.select',
                     'aten.transpose','aten.permute','aten.squeeze','aten.unsqueeze',
                     'aten.resize_','aten.resize_as_'}
            if '.'.join(name.split('.')[:2]) not in allowed:
                raise RuntimeError('Candidate invoked a non-metadata ATen entry point: '+name)
            return function(*args,**(kwargs or {}))

    candidate=imported('_capture_candidate',args.source)
    oracle=imported('_capture_input',args.leaf/'oracle.py')
    cases=[json.loads(line) for line in (args.leaf/'correctness.jsonl').read_text().splitlines() if line.strip()]
    timing=[json.loads(line) for line in (args.leaf/'timing.jsonl').read_text().splitlines() if line.strip()]
    largest={}
    for case in timing:
        spec=case.get('inputs',{}).get('spec',{})
        dtype=spec.get('dtype','unknown')
        size=math.prod(spec.get('shape',[]))
        if dtype not in largest or size>largest[dtype][0]:
            largest[dtype]=(size,case)
    selected=cases+[item[1] for item in largest.values()]
    original=JITFunction.run
    entries={}

    def capture(kernel,*values,**options):
        compiled=original(kernel,*values,**options)
        if compiled is not None and hasattr(compiled,'asm'):
            identity=(kernel.__name__,str(compiled.metadata))
            if identity not in entries:
                folder=args.output/('kernel-'+str(len(entries))); folder.mkdir()
                artifacts=[]
                for kind,value in compiled.asm.items():
                    if not isinstance(value,(str,bytes)): continue
                    path=folder/('kernel.'+kind)
                    path.write_bytes(value.encode() if isinstance(value,str) else value)
                    artifacts.append(dict(kind=kind,path=str(path),sha256=sha(path)))
                assert {'ttir','npubin'}.issubset({row['kind'] for row in artifacts})
                entries[identity]=dict(symbol=kernel.__name__,metadata=str(compiled.metadata),artifacts=artifacts)
        return compiled

    rows=[]
    JITFunction.run=capture
    try:
        for case in selected:
            inputs=oracle.gen_inputs(case,'npu:0')
            mode=CandidateOnly()
            with mode: candidate.run(**inputs)
            torch.npu.synchronize()
            rows.append(dict(case=case['name'],passed=True,aten_operations=mode.operations))
    finally:
        JITFunction.run=original
    assert entries,'No actual compiled artifacts were captured'
    result=dict(status='verified',source_sha256=sha(args.source),checker_sha256=sha(__file__),
                cases=rows,entries=list(entries.values()),passed=len(rows),total=len(rows),
                scope='Artifacts from actual candidate calls and ATen-path checks; use alongside execution-path review and full semantic tests')
    (args.output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],cases=len(rows),kernels=sorted({key[0] for key in entries}))))


if __name__=='__main__':
    main()
