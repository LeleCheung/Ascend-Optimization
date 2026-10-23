from pathlib import Path
import hashlib,importlib.util,json,time,torch,torch_npu
batch=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/remaining-five-20261023-v2')
source=batch/'knowledge/upsample-constexpr.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='91c51fa26b7a2e823b0de4a550eb04e30fec4c9bd7ae9b9d2d72f81a99946130'
parent=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/remaining-five-20261023-v2/results/upsample_bilinear2d/attempt-1/candidates/2054ca728198ae2a395addf604834a00f67f4e5352acc84adb93250457bee84b/main.py')
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='2054ca728198ae2a395addf604834a00f67f4e5352acc84adb93250457bee84b'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
original=load('constexpr_tile_parent',parent);candidate=load('constexpr_tile_candidate',source)
oracle=load('constexpr_tile_oracle',batch/'catalog/ops/upsample_bilinear2d/oracle.py')
cases=[json.loads(line) for line in (batch/'catalog/ops/upsample_bilinear2d/timing.jsonl').read_text().splitlines()][:2]
rows=[]
for case in cases:
 args=oracle.gen_inputs(case,'npu:0');expected=oracle.run(**args);torch.npu.synchronize()
 for name,module in [('parent',original),('constexpr_tile',candidate)]:
  actual=module.run(**args);torch.npu.synchronize()
  tol=case['tolerance'];torch.testing.assert_close(actual,expected,atol=tol['atol'],rtol=tol['rtol'])
  for _ in range(3):value=module.run(**args);torch.npu.synchronize()
  start=time.perf_counter()
  for _ in range(20):value=module.run(**args);torch.npu.synchronize()
  duration=(time.perf_counter()-start)*1000/20
  row=dict(case=case['name'],variant=name,walltime_ms=duration,correct=True)
  rows.append(row);print('FLAT_STORE',json.dumps(row),flush=True)
out=batch/'research/upsample_bilinear2d/constexpr-probe'
(out/'timings.json').write_text(json.dumps(dict(scope='Codex固定输入机制探索，非正式完整评测',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows),ensure_ascii=False,indent=2)+'\n')
