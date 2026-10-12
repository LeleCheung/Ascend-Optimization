#!/usr/bin/env bash
# 在已有容器中串行复验最终三个候选，不重新运行模型优化。
set -euo pipefail
root=${1:-/data/hanle/ascend-optimization}
repo="$root/Ascend-Optimization"
project="$repo/project/Ascend-Optimization-v1"
py=/usr/local/python3.11.15/bin/python3.11
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH="$repo/sources/kernelgen_server/client:$repo/sources/kernelgen_server:$repo/sources:${PYTHONPATH:-}"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
"$py" "$project/tools/environment/verify-flaggems-snapshot.py"
"$root/runtime/kg-controller/venv/bin/kg" --help >/dev/null
"$root/runtime/kg-controller/claude-native" --version
for operator in amin matmul_bias_activation narrow_copy; do
  case "$operator" in
    amin) candidate=direct-axis-native-v5-20261010.py; port=19655; scope=device_kernel ;;
    matmul_bias_activation) candidate=profiling-dual-dispatch-v8-20261010.py; port=19655; scope=device_kernel ;;
    narrow_copy) candidate=profile-small-dma-hybrid-20261010.py; port=19657; scope=device_task ;;
  esac
  "$py" "$project/tools/evaluation/evaluate-operator.py" \
    --operator "$operator" --source "$project/operators/$operator/candidates/$candidate" \
    --server "http://127.0.0.1:$port" --timing-scope "$scope" \
    --output "$project/operators/$operator/reports/directory-migration-20261012" \
    --label optimized --timeout 1800
done
echo '三个最终候选的目录迁移复验全部完成。'
