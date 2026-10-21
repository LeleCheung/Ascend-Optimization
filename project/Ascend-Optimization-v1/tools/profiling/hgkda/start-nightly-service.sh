#!/usr/bin/env bash
# 910B宿主执行；独立Native服务显式绑定固定FlagGems，不重启已有服务。
set -euo pipefail
night_batch=${1:?请指定批次名}
[[ "$night_batch" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
night_root=${HGKDA_ROOT:-/data/hanle/ascend-optimization}
night_repo="$night_root/Ascend-Optimization"
night_project="$night_repo/project/Ascend-Optimization-v1"
night_port=${HGKDA_PORT:-19659}
night_container=${HGKDA_CONTAINER:-tle_yy}
night_session="hanle-kgs-$night_batch"
night_runtime="$night_root/runs/$night_batch"
mkdir -p "$night_runtime"
if tmux has-session -t "$night_session" 2>/dev/null; then
  echo '服务会话已存在，拒绝重复启动'; exit 2
fi
python3 - "$night_port" <<'PY'
import socket,sys
s=socket.socket();s.settimeout(2)
if s.connect_ex(('127.0.0.1',int(sys.argv[1])))==0:
    raise SystemExit('端口已被占用，拒绝重复启动')
PY
printf -v night_command '%q ' docker exec -e "PYTHONPATH=$night_repo/third_party/FlagGems/src" \
  "$night_container" bash "$night_project/tools/evaluation/start-fixed-master-kgs.sh" \
  "$night_repo" "$night_repo/third_party/FlagGems" "$night_port" 7 "$night_runtime/profiles"
printf -v night_command '%s >%q 2>&1' "$night_command" "$night_runtime/kgs.log"
tmux new-session -d -s "$night_session" "$night_command"
echo "已启动 $night_session，端口 $night_port；请等 /status 健康后启动队列。"
