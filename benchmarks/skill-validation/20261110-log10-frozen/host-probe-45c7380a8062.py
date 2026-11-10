import fcntl,hashlib,json,shutil,subprocess,time
from pathlib import Path
r=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1');digest='45c7380a8062a27eba4666c2c71e4970b6d3f52b6d8227ba872a21830266c350'
preflight=r/"attempts"/digest/"preflight.json"
for _ in range(180):
 if preflight.exists():break
 time.sleep(5)
assert json.loads(preflight.read_bytes())["status"]=="passed"
with Path('/data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock').open("a") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 subprocess.run(['docker', 'exec', 'tle_yy', 'bash', '-c', 'source /usr/local/Ascend/cann-9.0.0/set_env.sh\nexport ASCEND_RT_VISIBLE_DEVICES=7\nexport OMP_NUM_THREADS=1\nexport PYTHONDONTWRITEBYTECODE=1\nexport TRITON_CACHE_DIR=/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1/attempts/45c7380a8062a27eba4666c2c71e4970b6d3f52b6d8227ba872a21830266c350/triton-cache\nexec /usr/local/python3.11.15/bin/python3.11 /data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v42/tools/skill_validation/launch_probe.py --source /data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1/attempts/45c7380a8062a27eba4666c2c71e4970b6d3f52b6d8227ba872a21830266c350/candidate.py --leaf /data/hanle/ascend-optimization/runs/independent-regression-20261109-v3/operators/log10_/supplementary-catalog --case log10_-timing-0 --case log10_-timing-3 --output /data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1/launch-probe-45c7380a8062.json --exercise-hooks'],check=True)
out=Path('/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-frozen-v1/launch-probe-45c7380a8062.json')
assert json.loads(out.read_bytes())["source_sha256"]==digest
assert hashlib.sha256((r/"workspace/candidate.py").read_bytes()).hexdigest()==digest
shutil.copyfile(out,r/"workspace/launch-probe.json")
(r/("host-probe-"+digest[:12]+"-completed.json")).write_text(json.dumps(dict(status="completed",source_sha256=digest)))
