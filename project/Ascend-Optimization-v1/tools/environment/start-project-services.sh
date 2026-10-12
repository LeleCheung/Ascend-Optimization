#!/usr/bin/env bash
# 在宿主机运行；只创建本项目的两个 tmux 会话，不停止已有服务。
set -euo pipefail
root=${1:-/data/hanle/ascend-optimization}
container=${2:-tle_yy}
device=${3:-7}
case "$device" in 6|7) ;; *) echo '本项目获准使用的物理卡为 6、7' >&2; exit 2;; esac
repo="$root/Ascend-Optimization"
project="$repo/project/Ascend-Optimization-v1"
narrow="$root/runtime/flaggems-narrow-device"
services="$root/runs/services"
mkdir -p "$services"
test -f "$repo/manifests/flaggems-master-files.json"
if ! test -d "$narrow"; then
  python3 "$project/tools/environment/prepare-narrow-runtime.py" --repo "$repo" --output "$narrow"
fi
for port in 19655 19657; do
  docker exec "$container" python3 -c 'import socket,sys
s=socket.socket();s.settimeout(2)
if s.connect_ex(("127.0.0.1",int(sys.argv[1])))==0:
 raise SystemExit("端口已占用，拒绝启动重复服务")' "$port"
done
for session in hanle-kgs-master hanle-kgs-narrow; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "会话已存在，拒绝覆盖：$session" >&2; exit 2
  fi
done
launch() {
  local session=$1 gems=$2 port=$3 command
  printf -v command '%q ' docker exec "$container" bash \
    "$project/tools/evaluation/start-fixed-master-kgs.sh" "$repo" "$gems" "$port" "$device" "$services/profiles-$port"
  printf -v command '%s >>%q 2>&1' "$command" "$services/kgs-$port.log"
  tmux new-session -d -s "$session" "$command"
}
launch hanle-kgs-master "$repo/third_party/FlagGems" 19655
launch hanle-kgs-narrow "$narrow" 19657
echo '已启动本项目两个 KGS tmux 会话，请检查 /status 和 runs/services 日志。'
