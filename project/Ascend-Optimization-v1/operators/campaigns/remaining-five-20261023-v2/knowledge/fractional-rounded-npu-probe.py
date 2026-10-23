from pathlib import Path
import importlib.util,json,hashlib
root=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
batch=root/'operators/campaigns/remaining-five-20261023-v2'
source=batch/'knowledge/fractional-rounded-intervals.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='f3c03df479bba4c5fed1f81806cb105779dac7011c57dbeda429721895fd012c'
out=batch/'research/fractional_max_pool3d/rounded-interval-probe'
checks=[]
for filename,name in [('check-remaining-five.py','semantics.json'),('check-remaining-special.py','special.json')]:
 spec=importlib.util.spec_from_file_location('rounded_'+name.replace('.','_'),root/'tools/profiling/hgkda'/filename)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 code=module.check('fractional_max_pool3d',source,out/name)
 data=json.loads((out/name).read_text())
 print('CHECK',json.dumps(dict(name=name,status=data['status'],passed=data['passed'],total=data['total'],failures=[r for r in data['cases'] if r['status']!='passed']),ensure_ascii=False),flush=True)
 checks.append(code)
raise SystemExit(int(any(checks)))
