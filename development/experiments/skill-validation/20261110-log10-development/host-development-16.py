import json,os,shutil,subprocess,time
from pathlib import Path
run=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2');digest='d519b459c915e43f123e5fd15273adf91209737a4677a7584e773ea6ac67d39f'
for _ in range(180):
 if list(run.glob("host-evaluate-"+digest[:12]+"*-completed.json")):break
 time.sleep(5)
assert list(run.glob("host-evaluate-"+digest[:12]+"*-completed.json")),"前一正式评测尚未完整收尾"
assert json.loads((run/"workspace/performance-feedback.json").read_bytes())["result"]["source_sha256"]==digest
r=subprocess.run(['/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python', '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v30/tools/skill_validation/agent_runner.py', '--config', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/config.json', '--model-settings', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/model-settings.json', '--invocation', 'development-16', '--max-turns', '40'],env=dict(os.environ)|{'PYTHONPATH': '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v30/src', 'PYTHONDONTWRITEBYTECODE': '1'})
assert r.returncode==0,r.returncode
(run/'host-development-16-completed.json').write_text(json.dumps(dict(status="completed",source_sha256=digest)))
