from pathlib import Path
import importlib.util,json,hashlib
root=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
source=root/'operators/campaigns/scatter-segmented-20261023/knowledge/scatter-manual-segmented-v2-prototype.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='72c4d40d831f72028f60e079ac9aae9cf921c33f32b515c2c042c43cc795d97e'
checker=root/'tools/profiling/hgkda/check-endurance.py'
spec=importlib.util.spec_from_file_location('scatter_manual_full_gate',checker)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
output=root/'operators/campaigns/remaining-five-20261023-v2/research/scatter_add/manual-segmented-full-gate-v2-probe/semantics.json'
code=module.check('scatter_add',source,output)
data=json.loads(output.read_text())
print('FULL_GATE',json.dumps(dict(status=data['status'],passed=data['passed'],total=data['total'],timing=data['timing_input_correctness']),ensure_ascii=False),flush=True)
raise SystemExit(code)
