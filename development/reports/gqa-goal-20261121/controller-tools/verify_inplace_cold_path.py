"""Fault-inject archived host launch logic; never execute a device kernel."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('archived_inplace_host',a.source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    submits=[]
    class Kernel:
        def __getitem__(self,grid):
            return lambda *args,**kwargs:submits.append('submitted')
        def warmup(self,*args,**kwargs):
            raise RuntimeError('injected compile/cache setup exception')
    class Input:
        dtype='synthetic_dtype'
        def numel(self):return 1024
        def is_contiguous(self):return True
        def data_ptr(self):return 4096
        def get_device(self):return 0
    module._rad2deg_contig_kernel=Kernel()
    module.torch.npu.current_device=lambda:0
    module.run(Input())
    passed=len(submits)==1
    report=dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
                status='passed' if passed else 'failed',passed=int(passed),total=1,
                case='cold_inplace_compile_setup_exception',submissions=len(submits),expected=1,
                scope='Actual archived Python host control flow with mocked kernel dispatch; no device arithmetic performed')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
