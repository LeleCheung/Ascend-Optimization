import json,os,shutil,subprocess,time
from pathlib import Path
run=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2');digest='a4d38b964aa64e0964ad85aabcaa3aa25fa82ec89d7a4931f7036b08974dadb8'
r=subprocess.run(['/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python', '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v3/tools/skill_validation/collect_pair.py', '--config', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/config.json', '--source', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/workspace/candidate.py', '--case', 'log10_-timing-3', '--output', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/paired-a4d38b964aa6-log10_-timing-3'],env=dict(os.environ)|{'PYTHONPATH': '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v3/src', 'PYTHONDONTWRITEBYTECODE': '1'})
assert r.returncode==0,r.returncode
(run/'host-pair-a4d38b964aa6-log10_-timing-3-completed.json').write_text(json.dumps(dict(status="completed",source_sha256=digest)))
