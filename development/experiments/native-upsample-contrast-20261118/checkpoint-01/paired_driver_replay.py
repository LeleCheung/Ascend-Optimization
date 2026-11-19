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
    if spec.get('autotune_replay'):
        import triton
        replay=json.loads(Path(spec['autotune_replay']).read_bytes())
        assert replay['source_sha256']==spec['source_sha256']
        assert replay['case']==spec['case']
        for entry in replay['entries']:
            tuner=getattr(candidate,entry['symbol'])
            for item in entry['cache']:
                config=triton.Config(**item['config'])
                assert any(c.kwargs==config.kwargs and c.num_warps==config.num_warps
                           and c.num_stages==config.num_stages and c.num_ctas==config.num_ctas for c in tuner.configs)
                tuner.cache[tuple(item['key'])]=config
    # Resolve native autotuning before recording the selected specialization.
    for _ in range(spec.get('warmup_before_capture',0)):
        getattr(candidate,spec.get('entrypoint','run'))(**inputs)
    if spec.get('warmup_before_capture',0):torch.npu.synchronize()
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
        getattr(candidate,spec.get('entrypoint','run'))(**inputs);torch.npu.synchronize()
    finally:JITFunction.run=original
    autotune=[]
    for name,tuner in vars(candidate).items():
        if not hasattr(tuner,'configs') or not isinstance(getattr(tuner,'cache',None),dict):continue
        entries=[]
        for key,config in tuner.cache.items():
            assert not config.pre_hook, 'autotune replay cannot serialize hooks'
            entries.append(dict(key=list(key),config={k:getattr(config,k) for k in ['kwargs','num_warps','num_stages','num_ctas','maxnreg']}))
        autotune.append(dict(symbol=name,cache=entries))
    (Path(spec['ir_directory'])/'autotune-cache.json').write_text(json.dumps(dict(
        source_sha256=spec['source_sha256'],case=spec['case'],entries=autotune),indent=2)+'\n')
    value=dict(source_sha256=spec['source_sha256'],files=artifacts)
    (Path(spec['ir_directory'])/'manifest.json').write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
