import os,sys,torch,torch_npu,triton,importlib.util,traceback
os.environ['ASCEND_LAUNCH_BLOCKING']='1'
sys.path.insert(0,'/data/hanle/ascend-optimization/goal-20261010/FlagGems/src')
import flag_gems
spec=importlib.util.spec_from_file_location('candidate','profiling-k128-pipeline-v2-20261010.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.npu.set_device(0)
for shape,dtype in [((1,1,32),torch.float16),((15,160,1024),torch.float16),((384,384,384),torch.float16),((1,1,32),torch.float32),((384,384,384),torch.float32),((1,1,32),torch.bfloat16)]:
 M,N,K=shape
 a=torch.randn((M,K),device='npu',dtype=dtype);b=torch.randn((K,N),device='npu',dtype=dtype);bias=torch.randn(N,device='npu',dtype=dtype)
 ref=torch.relu(a.float()@b.float()+bias.float());torch.npu.synchronize()
 try:
  with flag_gems.use_gems():y=m.run(a,b,bias)
  torch.npu.synchronize()
  print('CHECK',shape,str(dtype),torch.isfinite(y).all().item(),(y.float()-ref).abs().max().item(),flush=True)
 except BaseException:
  traceback.print_exc();raise
