import os,sys,pathlib,traceback,inspect,subprocess
root=pathlib.Path('/data/hanle/ascend-optimization/goal-20261010/FlagGems')
candidate=pathlib.Path('flaggems-master-dim-adapter-v2-20261010.py').resolve()
os.chdir(root)
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'src'))
import torch,torch_npu,triton
import triton.backends.ascend.testing as t
print('DEVICE_ENV',{k:os.environ.get(k) for k in ['ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','GEMS_VENDOR']},flush=True)
print('TIMER',inspect.getsource(t.do_bench_npu),flush=True)
orig=t.do_bench_npu
counter=0
def timer(fn,*args,**kwargs):
 global counter
 counter+=1
 print('TIMER_START',counter,repr(fn),flush=True)
 try:
  y=fn();torch.npu.synchronize()
  print('WARM_OUTPUT',counter,y.shape,y.dtype,y.flatten()[:3].cpu().tolist(),flush=True)
  v=orig(fn,*args,**kwargs)
  print('TIMER_OK',counter,v,flush=True)
  return v
 except BaseException:
  print('TIMER_FAILED',counter,flush=True);traceback.print_exc();raise
t.do_bench_npu=timer
import benchmark.base as b
orig_latency=b.Benchmark.get_latency
def latency(self,op,*args,**kwargs):
 print('OP',str(op),flush=True)
 for a in args:
  if isinstance(a,torch.Tensor):
   print('INPUT',a.shape,a.dtype,a.stride(),a.flatten()[:3].cpu().tolist(),flush=True)
 return orig_latency(self,op,*args,**kwargs)
b.Benchmark.get_latency=latency
import pytest
args=['-q','-s','benchmark/test_amin.py','-m','amin','--level','core','--case-id','benchmark/test_amin.py::test_amin[dtype0]::core::float16::0','--record','json','--output',str(candidate.parent/'amin-benchmark.json'),'--override','amin:'+str(candidate)+':run']
print('PYTEST_ARGS',args,flush=True)
raise SystemExit(pytest.main(args))
