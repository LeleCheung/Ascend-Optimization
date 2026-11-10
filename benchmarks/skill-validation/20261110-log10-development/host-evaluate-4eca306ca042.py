import json,os,shutil,subprocess,time
from pathlib import Path
run=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2');digest='4eca306ca0421347bfaf7f8de79da7d76cb37e0e3a9459c4de806319dc8f3e36'
preflight=run/"attempts"/digest/"preflight.json"
for _ in range(180):
 if preflight.exists():break
 time.sleep(5)
assert json.loads(preflight.read_bytes())["status"]=="passed"
r=subprocess.run(['/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python', '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v11/tools/skill_validation/evaluate.py', '--config', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/config.json', '--source', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/workspace/candidate.py', '--review', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/source-reviews/4eca306ca0421347bfaf7f8de79da7d76cb37e0e3a9459c4de806319dc8f3e36.json', '--port', '19696'],env=dict(os.environ)|{'PYTHONPATH': '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v11/src', 'PYTHONDONTWRITEBYTECODE': '1'})
assert r.returncode==0,r.returncode
feedback=run/("formal-"+digest[:12])/"performance-feedback.json"
shutil.copyfile(feedback,run/"workspace/performance-feedback.json")
(run/'host-evaluate-4eca306ca042-completed.json').write_text(json.dumps(dict(status="completed",source_sha256=digest)))
