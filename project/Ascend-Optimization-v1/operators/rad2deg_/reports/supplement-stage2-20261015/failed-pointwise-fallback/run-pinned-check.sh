#!/usr/bin/env bash
set -euo pipefail
root=/data/hanle/ascend-optimization
exec 9>"$root/runs/physical-card-7-campaign.lock"
flock -w 3600 9
docker exec -i tle_yy bash -s <<'RUN'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH=/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src
/usr/local/python3.11.15/bin/python3.11 - <<'PY'
import hashlib,json,runpy,sys
from pathlib import Path
import flag_gems
repo=Path('/data/hanle/ascend-optimization/Ascend-Optimization')
assert Path(flag_gems.__file__).resolve().is_relative_to(repo/'third_party/FlagGems')
q=repo.parent/'runs/stage2-20261015-rad2deg-repair'
output=q/'extra-check-pinned.json'
assert not output.exists()
sys.argv=[str(q/'extra-check.py'),'--source',str(q/'candidate.py'),'--output',str(output)]
runpy.run_path(str(q/'extra-check.py'),run_name='__main__')
v=json.loads(output.read_text());v['flag_gems_source']=flag_gems.__file__
v['source_sha256']=hashlib.sha256((q/'candidate.py').read_bytes()).hexdigest()
output.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'pinned_extra_check':v['status'],'cases':len(v['cases']),'source':v['flag_gems_source']}))
PY
RUN
