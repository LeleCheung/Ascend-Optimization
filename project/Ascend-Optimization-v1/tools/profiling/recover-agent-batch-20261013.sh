#!/usr/bin/env bash
# 仅在确认910B重启后恢复本批任务；归档中断请求，再串行接续。
set -euo pipefail
expected_boot=${1:?传入已核实的宿主机boot_id}
root=/data/hanle/ascend-optimization
project="$root/Ascend-Optimization/project/Ascend-Optimization-v1"
test "$(cat /proc/sys/kernel/random/boot_id)" = "$expected_boot"
exec 8>"$root/runs/physical-card-7-campaign.lock"
flock -n 8 || { echo '物理卡7已有项目任务，拒绝重复恢复' >&2; exit 2; }
docker exec -i tle_yy bash -s -- "$expected_boot" <<'BASH'
set -euo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
py=/usr/local/python3.11.15/bin/python3.11
export KG_RUN_ROOT=/data/hanle/ascend-optimization/runs/20261012-agent/kg-runs
"$py" - "$p" "$1" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
project=Path(sys.argv[1]);expected_boot=sys.argv[2]
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==expected_boot
uptime=float(Path('/proc/uptime').read_text().split()[0])
boot_time=time.time()-uptime
sys.path.insert(0,str(project/'tools/profiling'))
from optimization_campaign import assert_card_idle
assert_card_idle()
for operator in ('matmul_bias_activation','amin'):
 closure=project/'operators'/operator/'reports/master-closure-20261012-agent'
 for label in ('master','master-compatible','kg-no-profile','kg-profile'):
  output=closure/'baselines'/label
  pending=[f for f in output.glob('*.request.json')
           if not f.with_name(f.name.replace('.request.json','.result.json')).exists()]
  if not pending:continue
  if any(f.stat().st_mtime>=boot_time for f in pending):
   raise RuntimeError('请求在本次开机后创建，不能按重启中断归档')
  archive=closure/('interrupted-boot-'+expected_boot)/label
  if archive.exists():raise RuntimeError('归档已存在，拒绝覆盖')
  files=[f for f in output.rglob('*') if f.is_file()]
  record={'reason':'宿主机重启中断，原请求保留后重新完整评测',
          'boot_id':expected_boot,'boot_time_epoch':boot_time,
          'pending_requests':[f.name for f in pending],
          'files':{f.relative_to(output).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
  archive.parent.mkdir(parents=True,exist_ok=True)
  output.rename(archive)
  log=closure/'baselines'/(label+'.log')
  if log.exists():log.rename(archive/'controller.log')
  (archive/'重启恢复记录.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  print('ARCHIVED_INTERRUPTED_EVALUATION',operator,label,expected_boot,flush=True)
PY
"$py" "$p/tools/profiling/finalize-agent-campaign.py" --operator matmul_bias_activation --prior-closure master-closure-20261012
"$py" "$p/tools/profiling/finalize-agent-campaign.py" --operator amin --prior-closure master-closure-20261010
"$py" "$p/tools/profiling/run-new-operator-closure-20261012.py" --batch 20261012-agent
BASH
echo 'RECOVERED_BATCH_COMPLETED'
