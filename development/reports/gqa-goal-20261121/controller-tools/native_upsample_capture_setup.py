"""Prepare isolated native evidence collection with the original function name."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

r=Path('/data/hanle/ascend-optimization/runs/native-upsample-contrast-20261118')
old=r.parent/'night-research-20261116'
with tarfile.open(r/'method.tar.gz') as t:t.extractall(r/'method',filter='data')
run=r/'native';run.mkdir();w=run/'workspace';w.mkdir()
for name in ['candidate.py','starter.py']:(w/name).write_bytes((r/'upstream.py').read_bytes())
shutil.copytree(old/'no-simulator/reference-catalog',run/'reference-catalog')
c=json.loads((old/'no-simulator/config.json').read_bytes())
c.update(run_directory=str(run),package=str(r/'method/hgkda'),context_kind='research_contrast',
    goal_kind='performance_recovery_with_formal_feedback',capture_entrypoint='upsample_bilinear2d',capture_warmup=1,
    kgs_server='http://127.0.0.1:19728',device_pythonpath='/data/hanle/ascend-optimization/Ascend-Optimization/development/dependencies/third_party/FlagGems/src')
(run/'config.json').write_text(json.dumps(c,indent=2))
print(run/'config.json')
