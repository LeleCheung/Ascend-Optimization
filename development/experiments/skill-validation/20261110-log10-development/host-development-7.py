import json,os,shutil,subprocess,time
from pathlib import Path
run=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2');digest='c3e1fd58e495d62ede3c947583710b9f82718b93d4fd6663e3892139752f4fba'
r=subprocess.run(['/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python', '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v10/tools/skill_validation/agent_runner.py', '--config', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/config.json', '--model-settings', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/model-settings.json', '--invocation', 'development-7', '--max-turns', '28'],env=dict(os.environ)|{'PYTHONPATH': '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v10/src', 'PYTHONDONTWRITEBYTECODE': '1'})
assert r.returncode==0,r.returncode
(run/'host-development-7-completed.json').write_text(json.dumps(dict(status="completed",source_sha256=digest)))
