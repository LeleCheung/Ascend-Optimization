#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
Usage:
  resume_simple_opt_chip_e2e.sh \
    --base DIR \
    --fib-repo DIR \
    --server URL \
    --target-hardware NAME \
    --trace-name NAME \
    --definition NAME \
    --workspace DIR \
    --log FILE \
    [--max-coder-sessions N]

Resume one existing SimpleOpt definition workspace without cleaning its ledger.
EOF
}

base=""
fib_repo=""
server=""
target_hardware=""
trace_name=""
definition=""
workspace=""
log_file=""
max_coder_sessions=3

while (($#)); do
  case "$1" in
    --base)
      base=$2
      shift 2
      ;;
    --fib-repo)
      fib_repo=$2
      shift 2
      ;;
    --server)
      server=$2
      shift 2
      ;;
    --target-hardware)
      target_hardware=$2
      shift 2
      ;;
    --trace-name)
      trace_name=$2
      shift 2
      ;;
    --definition)
      definition=$2
      shift 2
      ;;
    --workspace)
      workspace=$2
      shift 2
      ;;
    --log)
      log_file=$2
      shift 2
      ;;
    --max-coder-sessions)
      max_coder_sessions=$2
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

if [[ -z "$base" || -z "$fib_repo" || -z "$server" ||
      -z "$target_hardware" || -z "$trace_name" || -z "$definition" ||
      -z "$workspace" || -z "$log_file" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$max_coder_sessions" =~ ^[1-9][0-9]*$ ]]; then
  printf 'max-coder-sessions must be a positive integer\n' >&2
  exit 2
fi

kernelgen_repo="$base/kernelgen"
python_bin="$base/venv/bin/python"
runtime_bin="$base/runtime/claude/node_modules/.bin"
if [[ ! -x "$runtime_bin/claude" ]]; then
  # Compatibility with workspaces created before npm installation was
  # standardized. New deployments must use runtime/claude.
  legacy_runtime_bin="$base/runtime/bin"
  if [[ -x "$legacy_runtime_bin/claude" ]]; then
    runtime_bin="$legacy_runtime_bin"
  else
    printf 'missing Claude CLI; run tests/install_claude_cli.sh --base %s\n' \
      "$base" >&2
    exit 2
  fi
fi
mkdir -p "$(dirname "$log_file")"
exec >"$log_file" 2>&1

printf 'started_at=%s\n' "$(date -Iseconds)"
printf 'definition=%s\n' "$definition"
printf 'workspace=%s\n' "$workspace"
printf 'max_coder_sessions=%s\n' "$max_coder_sessions"

export PATH="$base/runtime/node-bin:$runtime_bin:$base/venv/bin:$PATH"
export PYTHONPATH="$base:$fib_repo${PYTHONPATH:+:$PYTHONPATH}"

# shellcheck source=/dev/null
source "$kernelgen_repo/env.sh"

"$python_bin" -u \
  "$kernelgen_repo/examples/simple_opt/run_example.py" \
  --definition-name "$definition" \
  --trace-root "$kernelgen_repo/trace_sets/$trace_name" \
  --workspace "$workspace" \
  --eval-server "$server" \
  --target-hardware "$target_hardware" \
  --model "$MODEL" \
  --max-coder-sessions "$max_coder_sessions"
resume_exit=$?
printf 'resume_exit=%s\n' "$resume_exit"
printf 'completed_at=%s\n' "$(date -Iseconds)"
exit "$resume_exit"
