#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start_eval_server.sh \
    --fib-repo DIR \
    --python PATH \
    --backend NAME \
    [--perf triton|profiler|walltime] \
    --port PORT \
    --log FILE \
    --pid-file FILE

Start a detached flashinfer-bench eval server. Device selection is inherited
from the caller (for example CUDA_VISIBLE_DEVICES or MUSA_VISIBLE_DEVICES).
Timing defaults to Triton do_bench; pass --perf profiler for the Ascend
torch_npu.profiler path.
EOF
}

fib_repo=""
python_bin=""
backend=""
perf="triton"
port=""
log_file=""
pid_file=""

while (($#)); do
  case "$1" in
    --fib-repo)
      fib_repo=$2
      shift 2
      ;;
    --python)
      python_bin=$2
      shift 2
      ;;
    --backend)
      backend=$2
      shift 2
      ;;
    --perf)
      perf=$2
      shift 2
      ;;
    --port)
      port=$2
      shift 2
      ;;
    --log)
      log_file=$2
      shift 2
      ;;
    --pid-file)
      pid_file=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$fib_repo" || -z "$python_bin" || -z "$backend" ||
      -z "$port" || -z "$log_file" || -z "$pid_file" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  printf 'port must be an integer between 1 and 65535\n' >&2
  exit 2
fi
if [[ "$perf" != "triton" && "$perf" != "profiler" && "$perf" != "walltime" ]]; then
  printf 'perf must be triton, profiler, or walltime\n' >&2
  exit 2
fi
if [[ ! -f "$fib_repo/eval_service/server.py" ]]; then
  printf 'missing eval server: %s\n' "$fib_repo/eval_service/server.py" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  printf 'python is not executable: %s\n' "$python_bin" >&2
  exit 2
fi

mkdir -p "$(dirname "$log_file")" "$(dirname "$pid_file")"

if [[ -s "$pid_file" ]]; then
  old_pid=$(<"$pid_file")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_state=$(ps -o stat= -p "$old_pid" 2>/dev/null || true)
    if [[ "$old_state" != Z* ]]; then
      printf 'server is already running with pid %s\n' "$old_pid" >&2
      exit 1
    fi
  fi
fi

export PYTHONPATH="$fib_repo${PYTHONPATH:+:$PYTHONPATH}"
{
  printf 'started_at=%s\n' "$(date -Iseconds)"
  printf 'fib_repo=%s\n' "$fib_repo"
  printf 'python=%s\n' "$python_bin"
  printf 'backend=%s\n' "$backend"
  printf 'port=%s\n' "$port"
  printf 'timing_strategy=%s\n' "$perf"
  for name in \
    CUDA_VISIBLE_DEVICES \
    HIP_VISIBLE_DEVICES \
    ROCR_VISIBLE_DEVICES \
    MTHREADS_VISIBLE_DEVICES \
    MUSA_VISIBLE_DEVICES \
    ASCEND_RT_VISIBLE_DEVICES; do
    if [[ -v "$name" ]]; then
      printf '%s=%s\n' "$name" "${!name}"
    fi
  done
} >"$log_file.launch.txt"

cd "$fib_repo"
nohup setsid "$python_bin" -u eval_service/server.py \
  --backend "$backend" \
  --perf "$perf" \
  --port "$port" \
  >"$log_file" 2>&1 </dev/null &
server_pid=$!
printf '%s\n' "$server_pid" >"$pid_file"
printf 'started pid=%s port=%s backend=%s timing=%s\n' \
  "$server_pid" "$port" "$backend" "$perf"
