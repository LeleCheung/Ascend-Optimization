import json,os,shutil,subprocess,time
from pathlib import Path
run=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2');digest='a73e15786d685eb73322e512340d98c43636222634d0a3f8a4a54222c505812b'
preflight=run/"attempts"/digest/"preflight.json"
for _ in range(180):
 if preflight.exists():break
 time.sleep(5)
assert json.loads(preflight.read_bytes())["status"]=="passed"
r=subprocess.run(['/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python', '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v8/tools/skill_validation/evaluate.py', '--config', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/config.json', '--source', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/workspace/candidate.py', '--review', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/source-reviews/a73e15786d685eb73322e512340d98c43636222634d0a3f8a4a54222c505812b.json', '--port', '19696'],env=dict(os.environ)|{'PYTHONPATH': '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v8/src', 'PYTHONDONTWRITEBYTECODE': '1'})
assert r.returncode==0,r.returncode
feedback=run/("formal-"+digest[:12])/"performance-feedback.json"
shutil.copyfile(feedback,run/"workspace/performance-feedback.json")
(run/'host-evaluate-a73e15786d68-completed.json').write_text(json.dumps(dict(status="completed",source_sha256=digest)))
