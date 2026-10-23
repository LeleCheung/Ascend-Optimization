from pathlib import Path
import importlib.util,json,hashlib
root=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
batch=root/'operators/campaigns/remaining-five-20261023-v2'
source=root/'operators/campaigns/scatter-segmented-20261023/knowledge/scatter-segmented-prototype.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='f0ac63a45b09a50860037edec5014f5547fd6bed9806bdd1d49e0603a23becaf'
checker=root/'tools/profiling/hgkda/check-scatter-dynamic-range.py'
spec=importlib.util.spec_from_file_location('scatter_prototype_checker',checker)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
output=batch/'research/scatter_add/segmented-prototype-probe/dynamic-range.json'
code=module.check(source,output)
print('CHECK_RESULT',json.dumps(json.loads(output.read_text()),ensure_ascii=False),flush=True)
raise SystemExit(code)
