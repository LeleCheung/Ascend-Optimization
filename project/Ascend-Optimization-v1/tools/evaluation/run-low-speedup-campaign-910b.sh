#!/usr/bin/env bash
# 串行运行固定 master 基线与 KG 两组实验；第三版本由分析后的修改另行复验。
set -euo pipefail

root=${1:?usage: run-low-speedup-campaign-910b.sh EXPERIMENT_ROOT [OPERATOR ...]}
shift
test $# -gt 0 || set -- amin matmul_bias_activation narrow_copy
repo="$root/Ascend-Optimization"
project="$repo/project/Ascend-Optimization-v1"
python=/usr/local/python3.11.15/bin/python3.11
server=${KG_SERVER_URL:-http://127.0.0.1:19655}

export KG_RUN_ROOT="$root/kg-runs"
export KG_SOURCE_ROOT="$repo/sources"
export KG_WALLTIME_PORT=${KG_WALLTIME_PORT:-19655}
export KG_PROFILE_PORT=${KG_PROFILE_PORT:-19655}
export KG_FOREGROUND=1
mkdir -p "$KG_RUN_ROOT" "$root/campaign-logs"
exec 9>"$root/campaign.lock"
flock -n 9 || { echo '此实验目录已有 campaign，拒绝并发启动'; exit 2; }

for op in "$@"; do
  case "$op" in
    amin) source_file=src/flag_gems/ops/amin.py ;;
    matmul_bias_activation) source_file=src/flag_gems/fused/matmul_bias_activation.py ;;
    narrow_copy) source_file=src/flag_gems/ops/narrow_copy.py ;;
    *) echo "未配置上游源码路径：$op" >&2; exit 2 ;;
  esac
  folder="$project/operators/$op"
  report="$folder/reports/master-20261010"
  seed="$folder/candidates/flaggems-master-20261010.py"
  mkdir -p "$folder/candidates" "$report"
  if ! test -f "$seed"; then
    git -C "$root/FlagGems" show "HEAD:$source_file" >"$seed"
    printf '\n\nrun = %s\n' "$op" >>"$seed"
  fi

  # 可复用已完成的基线；失败或尚未生成结果时，由评测工具明确报错。
  if ! "$python" "$project/tools/evaluation/evaluate-operator.py" \
      --operator "$op" --source "$seed" --label flaggems-master \
      --server "$server" --output "$report" --timeout 1800 \
      >"$root/campaign-logs/$op-baseline.log" 2>&1; then
    echo "$op 基线未通过，保留原始证据并继续下一对象"
    continue
  fi
  if ! "$python" - "$report/flaggems-master-1.result.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x.get('status') != 'PASSED':
    raise SystemExit('基线未通过完整评测')
value=x.get('geo_mean')
if not isinstance(value,(int,float)) or value <= 0:
    raise SystemExit('缺少有效加速比')
print('master baseline',value,flush=True)
if value > 0.8:
    raise SystemExit('master 复测已达门槛，保留版本差异记录；不自动进入优化')
PY
  then
    continue
  fi

  for arm in no-profile profile; do
    workspace="$KG_RUN_ROOT/$op-$arm-master-20261010"
    if test -e "$workspace"; then
      echo "已存在 $workspace，需检查状态后显式 resume；不覆盖"
      continue
    fi
    echo "START $op $arm"
    if bash "$project/tools/evaluation/run-kg-operator-910b.sh" \
        "$op" "$arm" "$arm-master-20261010" 2 "$seed" \
        >"$root/campaign-logs/$op-$arm.log" 2>&1; then
      echo "DONE $op $arm；最终性能以 ledger 和独立复验为准"
    else
      echo "FAILED $op $arm；保留 workspace 与日志"
    fi
  done
done
echo 'CAMPAIGN_DONE：基线和两组尝试结束；分析、第三版本和交付仍需完成'
