from pathlib import Path
import json,hashlib,tarfile
r=Path('/data/hanle/ascend-optimization/runs/gqa-attribution-20261117')
files=set()
files.add(r/'method.tar.gz')
files.add(r/'checker-source.tar.gz')
for p in r.iterdir():
    if p.is_file() and p.suffix in ['.py','.json','.md','.log'] and p.name!='historical-excerpts.json':files.add(p)
for arm in ['no-simulator','no-simulator-v2','with-simulator-v2']:
    base=r/arm
    for part in ['agent-calls','trace','source-reviews','evaluation-round-01','evaluation-round-02']:
        for p in (base/part).rglob('*'):
            if not p.is_file() or 'cache' in p.parts or '__pycache__' in p.parts:continue
            if p.suffix in ['.py','.json','.jsonl','.log','.md','.txt']:files.add(p)
    for name in ['config.json','controller-state.json','workspace/candidate.py','workspace/diagnosis.json','workspace/evidence-summary.json','workspace/simulator-summary.json','workspace/next-performance-feedback.json','workspace/ir-excerpt.txt']:
        p=base/name
        if p.is_file():files.add(p)
for part in ['capture','capture-after','paired']:
    for p in (r/part).rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ['cache','__pycache__']):continue
        if p.suffix in ['.json','.csv','.log','.ttir','.bcmlir','.source']:files.add(p)
for name in ['LICENSE','src/kernel_eval/utils/compare.py','src/kernel_eval/utils/thresholds.py','src/kernel_eval/utils/tensor_utils.py','src/kernel_eval/eval/evaluator.py','src/kernel_eval/eval/accuracy_eval.py']:
    files.add(r/'official'/name)
manifest={'schema':'hgkda.gqa-evidence-export/v1','source_root':str(r),'upstream_commit':'12180ce7e717e8eed7cf81083666b40f8b5dd765','files':[]}
for p in sorted(files):
    assert p.is_relative_to(r) and p.name!='model-settings.json'
    manifest['files'].append({'path':p.relative_to(r).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
dest=r/'evidence-export-manifest.json';dest.write_text(json.dumps(manifest,indent=2));files.add(dest)
with tarfile.open(r/'evidence-export.tar.gz','w:gz') as tar:
    for p in sorted(files):tar.add(p,arcname=p.relative_to(r).as_posix(),recursive=False)
print(json.dumps({'files':len(files),'bytes':sum(p.stat().st_size for p in files),'archive_bytes':(r/'evidence-export.tar.gz').stat().st_size}))
