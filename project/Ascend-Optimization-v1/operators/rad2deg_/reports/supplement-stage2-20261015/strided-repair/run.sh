#!/usr/bin/env bash
set -euo pipefail
root=/data/hanle/ascend-optimization
repo="$root/Ascend-Optimization"
p="$repo/project/Ascend-Optimization-v1"
q="$root/runs/stage2-20261015-rad2deg-strided"
exec 9>"$root/runs/physical-card-7-campaign.lock"
flock -w 3600 9 || { echo '等待上一批释放卡锁超时'; exit 2; }
python3 - <<'PY'
import json
from pathlib import Path
p=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/rad2deg_/reports/hgkda-stage2-20261015-rad2deg/campaign-state.json')
assert json.loads(p.read_text())['status']=='completed','上一批没有正常结束，不自动开始'
PY
docker exec -i tle_yy bash -s <<'RUN'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH=/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
q=/data/hanle/ascend-optimization/runs/stage2-20261015-rad2deg-strided
py=/usr/local/python3.11.15/bin/python3.11
"$py" "$q/extra-check.py" --source "$q/candidate.py" --output "$q/extra-check.json"
"$py" "$p/tools/evaluation/evaluate-operator.py" \
 --operator rad2deg_ --source "$q/candidate.py" --label hgkda-reviewed \
 --server http://127.0.0.1:19655 --repeats 2 \
 --output "$p/operators/rad2deg_/reports/reviewed-strided-stage2-20261015"
RUN
