"""同输入矩阵乘驱动；区分PyTorch的GEMM组件与完整融合算子。"""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
import torch
import torch_npu

HERE=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/hgkda/roofline');PROJECT=HERE.parents[3]

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('HGKDA_SIMULATOR')=='1' and os.environ.get('ASCEND_RT_VISIBLE_DEVICES') in ['0','7']
    spec=json.loads(a.spec.read_text());m,n,k=spec['shape'];dtype=getattr(torch,spec['dtype'])
    generator=torch.Generator().manual_seed(20261027)
    def random(shape):return torch.randn(shape,generator=generator).to(dtype=dtype,device='npu')
    x=random((m,k));y=random((k,n));b=random((n,));a.output.mkdir(parents=True,exist_ok=True)
    if spec['side']=='pytorch':
        result=torch.relu(torch.mm(x,y)+b);records=[]
    else:
        source=Path(spec['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==spec['source_sha256']
        candidate=load(source,'roof_matrix_candidate')
        sys.path.insert(0,str(HERE.parent/'mechanism'))
        helper=load(HERE.parent/'mechanism/probe.py','roof_matrix_capture')
        records,result=helper.capture_call(lambda _:candidate.run(x,y,b),None)
    torch.npu.synchronize();expected=torch.relu(torch.mm(x,y)+b)
    torch.testing.assert_close(result,expected,atol=1e-4*k,rtol={'float16':1e-3,'bfloat16':.016,'float32':1.3e-6}[spec['dtype']])
    compiled=[]
    for i,record in enumerate(records):
        paths=[];directory=a.output/('kernel-'+str(i));directory.mkdir(exist_ok=True)
        for kind,value in record['compiled'].asm.items():
            raw=value if isinstance(value,bytes) else str(value).encode();name=kind+('.bin' if isinstance(value,bytes) else '.txt');path=directory/name
            if path.exists():assert path.read_bytes()==raw
            else:path.write_bytes(raw)
            paths.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
        compiled.append(dict(symbol=record['jit'].fn.__name__,grid=record['grid'],artifacts=paths,metadata=str(record['compiled'].metadata),
            constants={key:value for key,value in record['bound'].items() if isinstance(value,(str,bool,int,float))}))
    (a.output/'driver.json').write_text(json.dumps(dict(status='passed',spec=spec,compiled=compiled),indent=2)+'\n')
    print(json.dumps(dict(status='passed',spec=spec)),flush=True)

if __name__=='__main__':main()
