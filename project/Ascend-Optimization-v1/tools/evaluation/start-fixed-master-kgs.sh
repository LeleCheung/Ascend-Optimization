#!/usr/bin/env bash
# 在已有 Ascend 容器内启动独立服务；设备与端口由调用者明确选择。
set -euo pipefail
repo=${1:?REPO FLAGGEMS_CHECKOUT PORT PHYSICAL_DEVICE PROFILE_ROOT}
flaggems=${2:?FLAGGEMS_CHECKOUT}
port=${3:?PORT}
physical_device=${4:?PHYSICAL_DEVICE}
profile_root=${5:?PROFILE_ROOT}
test -f "$repo/sources/kernelgen_server/kernelgen_server/server.py"
test -f "$flaggems/benchmark/test_narrow_copy.py"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="$physical_device"
export KGS_FLAGGEMS_ROOT="$flaggems"
export PYTHONPATH="$repo/sources/kernelgen_server/client:$repo/sources/kernelgen_server:$repo/sources:${PYTHONPATH:-}"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
exec /usr/local/python3.11.15/bin/python3.11 -u -m kernelgen_server.server \
 --host 127.0.0.1 --backend npu --timing walltime --max-workers 1 \
 --port "$port" --profile-artifact-root "$profile_root"
