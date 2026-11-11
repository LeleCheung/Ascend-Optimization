import json,os,shutil,subprocess,time
from pathlib import Path
run=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2');digest='0fefe838c58ceafa9c4c7ab856d9c0d9cdbd6bf4bdc2f14c4193358f899a0d72'
for _ in range(180):
 if list(run.glob("host-evaluate-"+digest[:12]+"*-completed.json")):break
 time.sleep(5)
assert list(run.glob("host-evaluate-"+digest[:12]+"*-completed.json")),"前一正式评测尚未完整收尾"
assert json.loads((run/"workspace/performance-feedback.json").read_bytes())["result"]["source_sha256"]==digest
for name in ["host-profile.json","launch-probe.json"]:
 for _ in range(180):
  p=run/"workspace"/name
  if p.exists() and json.loads(p.read_bytes()).get("source_sha256")==digest:break
  time.sleep(5)
 assert p.exists() and json.loads(p.read_bytes()).get("source_sha256")==digest,name
r=subprocess.run(['/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python', '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v24/tools/skill_validation/agent_runner.py', '--config', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/config.json', '--model-settings', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/model-settings.json', '--invocation', 'development-12', '--max-turns', '40'],env=dict(os.environ)|{'PYTHONPATH': '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v24/src', 'PYTHONDONTWRITEBYTECODE': '1'})
assert r.returncode==0,r.returncode
(run/'host-development-12-completed.json').write_text(json.dumps(dict(status="completed",source_sha256=digest)))
