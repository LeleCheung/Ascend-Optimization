#!/usr/bin/env bash
# Historical campaign harness; new experiments use kg run --batch-file.
set -u

usage() {
  cat <<'EOF'
Usage:
  run_batch_simple_opt_chip_e2e.sh \
    --base DIR \
    --fib-repo DIR \
    --server URL \
    --target-hardware NAME \
    [--max-workers N] \
    [--max-coder-sessions N] \
    [--run-name NAME] \
    [--reference-triton-dir DIR] \
    [--reference-triton-prompt-path FILE] \
    [--skip-legacy] \
    [--skip-v4] \
    [--legacy-definition NAME]... \
    [--v4-definition NAME]...

The base directory must contain:
  kernelgen/       KernelGen source tree
  runtime/claude/  Claude Code installed by npm
  venv/            Python environment with KernelGen runtime dependencies

When no definitions are supplied, the script uses the representative E2E set
selected for the seven-chip portability run.
EOF
}

base=""
fib_repo=""
server=""
target_hardware=""
max_workers=0
max_coder_sessions=3
run_name="batch_simple_opt_e2e"
reference_triton_dir=""
reference_triton_prompt_path=""
skip_legacy=0
skip_v4=0
legacy_definitions=()
v4_definitions=()

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
    --max-workers)
      max_workers=$2
      shift 2
      ;;
    --max-coder-sessions)
      max_coder_sessions=$2
      shift 2
      ;;
    --run-name)
      run_name=$2
      shift 2
      ;;
    --reference-triton-dir)
      reference_triton_dir=$2
      shift 2
      ;;
    --reference-triton-prompt-path)
      reference_triton_prompt_path=$2
      shift 2
      ;;
    --skip-legacy)
      skip_legacy=1
      shift
      ;;
    --skip-v4)
      skip_v4=1
      shift
      ;;
    --legacy-definition)
      legacy_definitions+=("$2")
      shift 2
      ;;
    --v4-definition)
      v4_definitions+=("$2")
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

if [[ -z "$base" || -z "$fib_repo" || -z "$server" || -z "$target_hardware" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$max_workers" =~ ^[0-9]+$ ]]; then
  printf 'max-workers must be a non-negative integer\n' >&2
  exit 2
fi
if [[ ! "$max_coder_sessions" =~ ^[1-9][0-9]*$ ]]; then
  printf 'max-coder-sessions must be a positive integer\n' >&2
  exit 2
fi
if [[ ! "$run_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'run-name may contain only letters, digits, dot, underscore, and dash\n' >&2
  exit 2
fi
if [[ -n "$reference_triton_dir" && ! -d "$reference_triton_dir" ]]; then
  printf 'reference-triton-dir is not a directory: %s\n' \
    "$reference_triton_dir" >&2
  exit 2
fi
if [[ -n "$reference_triton_prompt_path" ]]; then
  if [[ -z "$reference_triton_dir" ]]; then
    printf 'reference-triton-prompt-path requires reference-triton-dir\n' >&2
    exit 2
  fi
  if [[ ! -f "$reference_triton_prompt_path" ]]; then
    printf 'reference-triton-prompt-path is not a file: %s\n' \
      "$reference_triton_prompt_path" >&2
    exit 2
  fi
fi

if ((${#legacy_definitions[@]} == 0)); then
  legacy_definitions=(
    abl_t1_gelu
    abl_t1_matmul_basic
  )
fi

if ((${#v4_definitions[@]} == 0)); then
  v4_definitions=(
    flaggems_rsqrt
    flaggems_var
    flaggems_softplus_backward
    flaggems_max_unpool2d
    flaggems_max_pool3d_with_indices
    flaggems_uniform_
  )
fi

kernelgen_repo="$base/kernelgen"
python_bin="$base/venv/bin/python"
runtime_bin="$base/runtime/claude/node_modules/.bin"
run_root="$base/runs/$run_name"
orchestrator_log="$run_root/orchestrator.log"

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

mkdir -p "$run_root"
rm -f \
  "$run_root/exit" \
  "$run_root/server-status-after.json"
if ((skip_legacy == 0)); then
  rm -f "$run_root/legacy.exit"
fi
if ((skip_v4 == 0)); then
  rm -f "$run_root/v4.exit"
fi
exec >"$orchestrator_log" 2>&1

printf 'started_at=%s\n' "$(date -Iseconds)"
printf 'base=%s\n' "$base"
printf 'fib_repo=%s\n' "$fib_repo"
printf 'server=%s\n' "$server"
printf 'target_hardware=%s\n' "$target_hardware"
printf 'max_workers=%s\n' "$max_workers"
printf 'max_coder_sessions=%s\n' "$max_coder_sessions"
printf 'run_name=%s\n' "$run_name"
printf 'reference_triton_dir=%s\n' "$reference_triton_dir"
printf 'reference_triton_prompt_path=%s\n' "$reference_triton_prompt_path"
printf 'skip_legacy=%s skip_v4=%s\n' "$skip_legacy" "$skip_v4"
printf 'legacy_definitions=%s\n' "${legacy_definitions[*]}"
printf 'v4_definitions=%s\n' "${v4_definitions[*]}"

export PATH="$base/runtime/node-bin:$runtime_bin:$base/venv/bin:$PATH"
export PYTHONPATH="$base:$fib_repo${PYTHONPATH:+:$PYTHONPATH}"

# shellcheck source=/dev/null
source "$kernelgen_repo/env.sh"

"$python_bin" --version
claude --version
printf 'model=%s\n' "$MODEL"
curl -sS "$server/status"
printf '\n'

"$python_bin" - "$base/venv-installed.txt" <<'PY'
import importlib.metadata
import sys
from pathlib import Path

output = Path(sys.argv[1])
environment_root = Path(sys.prefix).resolve()
items = []
for distribution in importlib.metadata.distributions():
    try:
        location = Path(distribution.locate_file("")).resolve()
    except (OSError, RuntimeError):
        continue
    if not location.is_relative_to(environment_root):
        continue
    name = distribution.metadata.get("Name")
    if name:
        items.append(f"{name}=={distribution.version}")
output.write_text("\n".join(sorted(set(items), key=str.lower)) + "\n")
PY

if grep -Eiq '^(torch|triton)(==|$)' "$base/venv-installed.txt"; then
  printf 'refusing to run: isolated dependency list contains Torch or Triton\n' >&2
  printf '2\n' >"$run_root/exit"
  exit 2
fi

run_batch() {
  local trace_name=$1
  local workspace=$2
  shift 2
  local definitions=("$@")
  local args=()
  local name
  local reference_path
  local matched_references=0

  for name in "${definitions[@]}"; do
    args+=(--definition-name "$name")
    if [[ -n "$reference_triton_dir" ]]; then
      reference_path="$reference_triton_dir/$name.py"
      if [[ -f "$reference_path" ]]; then
        args+=(--reference-triton-path "$reference_path")
        ((matched_references += 1))
      fi
    fi
  done
  if ((matched_references > 0)) && [[ -n "$reference_triton_prompt_path" ]]; then
    args+=(--reference-triton-prompt-path "$reference_triton_prompt_path")
  fi
  printf 'matched reference Triton sources for %s/%s definitions\n' \
    "$matched_references" "${#definitions[@]}"

  "$python_bin" -u \
    -m kernelgen.cli.legacy_batch_simple_opt \
    "${args[@]}" \
    --trace-root "$kernelgen_repo/trace_sets/$trace_name" \
    --workspace "$workspace" \
    --eval-server "$server" \
    --target-hardware "$target_hardware" \
    --model "$MODEL" \
    --max-workers "$max_workers" \
    --max-coder-sessions "$max_coder_sessions" \
    --clean
}

if ((skip_legacy == 0)); then
  run_batch \
    unified-trace-akg-bench-lite \
    "$run_root/legacy" \
    "${legacy_definitions[@]}" \
    >"$run_root/legacy.log" 2>&1
  legacy_exit=$?
  printf '%s\n' "$legacy_exit" >"$run_root/legacy.exit"
else
  legacy_exit=$(cat "$run_root/legacy.exit" 2>/dev/null || printf '0')
fi

if ((skip_v4 == 0)); then
  run_batch \
    unified-trace-flaggems-v4 \
    "$run_root/v4" \
    "${v4_definitions[@]}" \
    >"$run_root/v4.log" 2>&1
  v4_exit=$?
  printf '%s\n' "$v4_exit" >"$run_root/v4.exit"
else
  v4_exit=$(cat "$run_root/v4.exit" 2>/dev/null || printf '0')
fi

curl -sS "$server/status" >"$run_root/server-status-after.json"
printf '\n' >>"$run_root/server-status-after.json"

final_exit=0
if ((legacy_exit != 0 || v4_exit != 0)); then
  final_exit=1
fi
printf '%s\n' "$final_exit" >"$run_root/exit"
printf 'legacy_exit=%s v4_exit=%s final_exit=%s\n' \
  "$legacy_exit" "$v4_exit" "$final_exit"
printf 'completed_at=%s\n' "$(date -Iseconds)"
exit "$final_exit"
