import json,os,shutil,subprocess,time
from pathlib import Path
run=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1');digest='45c7380a8062a27eba4666c2c71e4970b6d3f52b6d8227ba872a21830266c350'
for _ in range(180):
 if list(run.glob("host-evaluate-"+digest[:12]+"*-completed.json")):break
 time.sleep(5)
assert list(run.glob("host-evaluate-"+digest[:12]+"*-completed.json")),"前一正式评测尚未完整收尾"
assert json.loads((run/"workspace/performance-feedback.json").read_bytes())["result"]["source_sha256"]==digest
r=subprocess.run(['/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python', '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v42/tools/skill_validation/agent_runner.py', '--config', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1/config.json', '--model-settings', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1/model-settings.json', '--invocation', 'frozen-07', '--max-turns', '40'],env=dict(os.environ)|{'PYTHONPATH': '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v42/src', 'PYTHONDONTWRITEBYTECODE': '1'})
assert r.returncode==0,r.returncode
(run/'host-frozen-07-completed.json').write_text(json.dumps(dict(status="completed",source_sha256=digest)))
