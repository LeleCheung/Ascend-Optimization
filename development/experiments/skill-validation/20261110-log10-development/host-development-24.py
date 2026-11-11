import json,os,shutil,subprocess,time
from pathlib import Path
run=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2');digest='8600ce07031842c6bed467ea5ebe1f0a66d265e864b171e4176c1f0b6b318324'
r=subprocess.run(['/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python', '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v42/tools/skill_validation/agent_runner.py', '--config', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/config.json', '--model-settings', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/model-settings.json', '--invocation', 'development-24', '--max-turns', '40'],env=dict(os.environ)|{'PYTHONPATH': '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v42/src', 'PYTHONDONTWRITEBYTECODE': '1'})
assert r.returncode==0,r.returncode
(run/'host-development-24-completed.json').write_text(json.dumps(dict(status="completed",source_sha256=digest)))
