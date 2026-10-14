#!/usr/bin/env bash
set -euo pipefail
root=/data/hanle/ascend-optimization
repo="$root/Ascend-Optimization"
project="$repo/project/Ascend-Optimization-v1"
batch=20261014-logadd-method-retry
exec 8>"$root/runs/physical-card-7-campaign.lock"
echo 'WAIT_CURRENT_FOUR_QUEUE'
flock 8
echo 'ACQUIRED_PHYSICAL_CARD_7'
test "$(git -C "$repo" rev-parse HEAD)" = 514c343bd383debb43d15c0628b52f4f11cdb826
docker exec -i tle_yy bash -s <<'RUN'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
py=/usr/local/python3.11.15/bin/python3.11
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
r=/data/hanle/ascend-optimization/runs/20261014-logadd-method-retry
"$py" - <<'PY'
import json
from pathlib import Path
p=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
for op in ['logaddexp2','amin','matmul_bias_activation','narrow_copy']:
 state=p/'operators'/op/'reports/profiling-agent-20261014-four-direct-method/campaign-state.json'
 assert state.is_file() and json.loads(state.read_text())['status']=='completed',op
PY
"$py" "$p/tools/profiling/optimization_campaign.py" \
 --operator logaddexp2 --seed "$p/operators/logaddexp2/candidates/flaggems-master-20261012-agent.py" \
 --output "$p/operators/logaddexp2/reports/profiling-agent-20261014-logadd-method-retry-method" \
 --server http://127.0.0.1:19655 --timing-scope device_kernel --rounds 3 --profile-cases 2 \
 --knowledge "$r/knowledge.md" --plain-json --max-output-tokens 16384 --provider-thinking disabled
"$py" "$p/tools/profiling/four_operator_workflow.py" --batch 20261014-logadd-method-retry \
 --operators logaddexp2 --mode method --replay-batch 20261014-four --report-only
RUN
