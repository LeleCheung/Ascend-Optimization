import json
import shutil
from pathlib import Path

root = Path('/data/hanle/ascend-optimization/runtime/kg-controller')
source = Path('/data/hanle/ascend-optimization/sources/kernelgen_server-6.5.0-archive/kernelgen_server-v6.5.0/data/kernelgenbench')
target = root / 'catalog-square'
operator = 'kernelgenbench_square'
relative = Path('ops/pointwise') / operator

manifest = json.loads((source / 'manifest.json').read_text())
entry = next(item for item in manifest['operators'] if item['name'] == operator)
manifest['operators'] = [entry]
manifest['counts'] = {
    'operators': 1,
    'correctness_workloads': entry['num_correctness_workloads'],
    'timing_workloads': entry['num_timing_workloads'],
}
manifest['name'] = 'kernelgenbench-square-private'
target.mkdir(exist_ok=True)
(target / relative).mkdir(parents=True, exist_ok=True)
(target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
for name in ('definition.json', 'oracle.py', 'correctness.jsonl', 'timing.jsonl',
             'correctness_full.jsonl', 'timing_full.jsonl'):
    shutil.copy2(source / relative / name, target / relative / name)
print(target)
