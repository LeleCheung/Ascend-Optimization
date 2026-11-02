import os,json,hashlib,importlib.util
from pathlib import Path
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import torch,torch_npu,triton
import triton.language.standard as standard
for name,existing in [('_elementwise_max_default','_elementwise_max'),('_elementwise_min_default','_elementwise_min')]:
 if not hasattr(standard,name):setattr(standard,name,getattr(standard,existing))
def forbidden(*args,**kwargs):raise RuntimeError('CPU结构诊断禁止NPU')
torch.npu._lazy_init=forbidden
class NoRuntime:
 def __getattr__(self,name):forbidden()
triton.runtime.driver.set_active(NoRuntime())
out=Path(__file__).parent;b=out.parents[2]
parent=b/'proposals/linalg_inv_ex/codex-compensated-dynamic-v5/main.py';text=parent.read_text();a=text.index('@triton.jit\ndef _corrected_div');z=text.index('@triton.jit\ndef _leftlooking');text=text[:a]+'@triton.jit\ndef _corrected_div(a,b):\n    return a / b\n\n\n'+text[z:];variant=out/'variant.py';variant.write_text(text)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('inverse_cpu_structure',variant);assert type(m._leftlooking).__name__=='InterpretedFunction'
leaf=b/'original-catalog/ops/linalg_inv_ex';oracle=load('inverse_cpu_oracle',leaf/'oracle.py');rows=[]
for ctx in [json.loads(line) for line in (leaf/'correctness.jsonl').read_text().splitlines()]:
 try:
  args=oracle.gen_inputs(ctx,'cpu');actual=m.run(**args);reference=oracle.run(**args)
  row=dict(case=ctx['name'],input=args['A'].tolist(),actual=actual.inverse.tolist(),reference=reference.inverse.tolist(),info=actual.info.tolist(),reference_info=reference.info.tolist())
  try:
   torch.testing.assert_close(actual.inverse,reference.inverse,atol=ctx['tolerance']['atol'],rtol=ctx['tolerance']['rtol'],equal_nan=True);torch.testing.assert_close(actual.info,reference.info);row['passed']=True
  except AssertionError as e:row.update(passed=False,error=str(e)[:1000])
 except Exception as e:row=dict(case=ctx['name'],error=type(e).__name__+': '+str(e)[:2000],passed=False)
 rows.append(row)
value=dict(status='cpu_interpreter_structure_diagnostic',parent_source_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),variant_sha256=hashlib.sha256(variant.read_bytes()).hexdigest(),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,device_executed=False,scope='CPU解释LU与求解结构；解释器缺少默认max/min别名，在本进程添加有限值等价别名。仅CPU诊断把商修正替换为CPU精确普通商，避开解释器bitwise_or兼容问题；不是原NPU候选成绩。')
(out/'result.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([dict(case=r['case'],passed=r['passed'],error=r.get('error')) for r in rows]))
