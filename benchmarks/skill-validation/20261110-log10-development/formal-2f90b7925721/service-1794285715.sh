source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
export KGS_FLAGGEMS_ROOT=/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems
export PYTHONPATH=/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server/client:/data/hanle/ascend-optimization/Ascend-Optimization/sources/kernelgen_server:/data/hanle/ascend-optimization/Ascend-Optimization/sources
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export TRITON_CACHE_DIR=/data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/formal-2f90b7925721/service-triton-cache
export PYTHONDONTWRITEBYTECODE=1
exec /usr/local/python3.11.15/bin/python3.11 -u -m kernelgen_server.server --host 127.0.0.1 --backend npu --timing walltime --max-workers 1 --port 19696 --profile-artifact-root /data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/formal-2f90b7925721/profiles
