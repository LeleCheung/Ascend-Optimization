#!/usr/bin/env bash
# 等矩阵乘、amin及筛选完成后启动新算子闭环；仅使用获准物理卡7。
set -euo pipefail
r=/data/hanle/ascend-optimization
python3 - <<'PY'
import json,pathlib,time
root=pathlib.Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
deadline=time.monotonic()+24*3600
while time.monotonic()<deadline:
 states=[]
 for operator in ('matmul_bias_activation','amin'):
  path=root/'operators'/operator/'reports/profiling-agent-20261012-agent/campaign-state.json'
  states.append(json.loads(path.read_text()).get('status') if path.exists() else None)
 if 'needs_attention' in states:raise SystemExit('campaign needs attention')
 if states==['completed','completed'] and (root/'operators/next-operator-screen-20261012-agent.json').exists():break
 time.sleep(30)
else:raise SystemExit('campaign wait timeout')
PY
exec 8>"$r/runs/physical-card-7-campaign.lock"
flock 8
docker exec -i tle_yy bash -s <<'BASH'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
export KG_RUN_ROOT=/data/hanle/ascend-optimization/runs/20261012-agent/kg-runs
"$py" - <<'PY'
import json,pathlib,urllib.request
root=pathlib.Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
for operator in ('matmul_bias_activation','amin'):
 state=json.loads((root/'operators'/operator/'reports/profiling-agent-20261012-agent/campaign-state.json').read_text())
 assert state['status']=='completed',state['status']
for port in (19655,19657):
 d=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/status'))['scheduler']
 assert d['active']==d['waiting']==d['broken']==0 and d['healthy']==1,d
PY
"$py" "$p/tools/profiling/run-new-operator-closure-20261012.py" --batch 20261012-agent
BASH
