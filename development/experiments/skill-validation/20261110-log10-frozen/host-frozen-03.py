import json,os,shutil,subprocess,time
from pathlib import Path
run=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1');digest='5ac2254b6040ac4615ce440b7d74fa461149b76417e896cb100687f2da85c6f8'
r=subprocess.run(['/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python', '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v42/tools/skill_validation/agent_runner.py', '--config', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1/config.json', '--model-settings', '/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1/model-settings.json', '--invocation', 'frozen-03', '--max-turns', '40'],env=dict(os.environ)|{'PYTHONPATH': '/data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v42/src', 'PYTHONDONTWRITEBYTECODE': '1'})
assert r.returncode==0,r.returncode
(run/'host-frozen-03-completed.json').write_text(json.dumps(dict(status="completed",source_sha256=digest)))
