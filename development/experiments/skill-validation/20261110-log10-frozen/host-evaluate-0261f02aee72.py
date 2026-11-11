import json,os,shutil,subprocess,time
from pathlib import Path
run=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1');digest='0261f02aee725d0866a9342f5e6d8ce62d5e7ab02f2b75ce3f824b679b898d3b'
preflight=run/"attempts"/digest/"preflight.json"
for _ in range(180):
 if preflight.exists():break
 time.sleep(5)
assert json.loads(preflight.read_bytes())["status"]=="passed"
r=subprocess.run(['/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python', '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v42/tools/skill_validation/evaluate.py', '--config', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1/config.json', '--source', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1/workspace/candidate.py', '--review', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1/source-reviews/0261f02aee725d0866a9342f5e6d8ce62d5e7ab02f2b75ce3f824b679b898d3b.json', '--port', '19696'],env=dict(os.environ)|{'PYTHONPATH': '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v42/src', 'PYTHONDONTWRITEBYTECODE': '1'})
assert r.returncode==0,r.returncode
feedback=run/("formal-"+digest[:12])/"performance-feedback.json"
shutil.copyfile(feedback,run/"workspace/performance-feedback.json")
(run/'host-evaluate-0261f02aee72-completed.json').write_text(json.dumps(dict(status="completed",source_sha256=digest)))
