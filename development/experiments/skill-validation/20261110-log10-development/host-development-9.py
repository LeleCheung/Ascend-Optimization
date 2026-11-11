import json,os,shutil,subprocess,time
from pathlib import Path
run=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2');digest='4eca306ca0421347bfaf7f8de79da7d76cb37e0e3a9459c4de806319dc8f3e36'
for _ in range(180):
 if list(run.glob("host-evaluate-"+digest[:12]+"*-completed.json")):break
 time.sleep(5)
assert list(run.glob("host-evaluate-"+digest[:12]+"*-completed.json")),"前一正式评测尚未完整收尾"
assert json.loads((run/"workspace/performance-feedback.json").read_bytes())["result"]["source_sha256"]==digest
r=subprocess.run(['/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python', '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v14/tools/skill_validation/agent_runner.py', '--config', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/config.json', '--model-settings', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/model-settings.json', '--invocation', 'development-9', '--max-turns', '28'],env=dict(os.environ)|{'PYTHONPATH': '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v14/src', 'PYTHONDONTWRITEBYTECODE': '1'})
assert r.returncode==0,r.returncode
(run/'host-development-9-completed.json').write_text(json.dumps(dict(status="completed",source_sha256=digest)))
