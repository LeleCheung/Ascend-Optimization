#!/usr/bin/env bash
set -euo pipefail

script_dir=$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)
# shellcheck source=tests/multi_device_batch_lib.sh
source "$script_dir/multi_device_batch_lib.sh"

usage() {
  cat <<'EOF'
Usage:
  collect_multi_device_batch_results.sh \
    --output DIR \
    [--run-name NAME] \
    [--inventory FILE] \
    [--identity FILE] \
    [--device NAME]... \
    [--extract]

Stream each complete run directory into a local .tar.gz. JumpServer banners are
ignored by extracting only the base64 payload between explicit markers.
EOF
}

output=""
run_name="batch_simple_opt_v5_overlap15_20260730"
extract=0
devices=()

while (($#)); do
  case "$1" in
    --output)
      output=$2
      shift 2
      ;;
    --run-name)
      run_name=$2
      shift 2
      ;;
    --inventory)
      MULTI_DEVICE_INVENTORY=$2
      shift 2
      ;;
    --identity)
      MULTI_DEVICE_IDENTITY=$2
      shift 2
      ;;
    --device)
      devices+=("$2")
      shift 2
      ;;
    --extract)
      extract=1
      shift
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

[[ -n "$output" ]] || {
  usage >&2
  exit 2
}
[[ "$run_name" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'invalid run name: %s\n' "$run_name" >&2
  exit 2
}
[[ -r "$MULTI_DEVICE_INVENTORY" ]] || {
  printf 'inventory is not readable: %s\n' "$MULTI_DEVICE_INVENTORY" >&2
  exit 2
}
[[ -r "$MULTI_DEVICE_IDENTITY" ]] || {
  printf 'SSH identity is not readable: %s\n' "$MULTI_DEVICE_IDENTITY" >&2
  exit 2
}
mkdir -p "$output"
output=$(cd -- "$output" && pwd)

records=$(multi_device_inventory_records) || {
  printf 'invalid inventory section: %s\n' "$MULTI_DEVICE_INVENTORY" >&2
  exit 2
}
while IFS='|' read -r \
  name mode address ssh_port container deploy_base server_port cards \
  jump_login; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  multi_device_validate_record \
    "$name" "$mode" "$address" "$ssh_port" "$container" \
    "$deploy_base" "$server_port" "$jump_login" || {
      printf 'invalid inventory record for %s\n' "$name" >&2
      exit 2
    }
  multi_device_selected "$name" "${devices[@]}" || continue

  archive="$output/${name}-${run_name}.tar.gz"
  partial="$archive.partial"
  [[ ! -e "$archive" && ! -e "$partial" ]] || {
    printf 'refusing to overwrite: %s\n' "$archive" >&2
    exit 2
  }
  printf 'collecting %s -> %s\n' "$name" "$archive"

  {
    printf 'name=%q\n' "$name"
    printf 'deploy_base=%q\n' "$deploy_base"
    printf 'run_name=%q\n' "$run_name"
    cat <<'REMOTE'
set -euo pipefail
run_root="$deploy_base/runs/${run_name}_${name}"
if [[ ! -d "$run_root" ]]; then
  run_root="$deploy_base/runs/$run_name"
fi
[[ -d "$run_root" ]] || {
  printf 'missing run root: %s\n' "$run_root" >&2
  exit 3
}
printf '__KERNELGEN_ARCHIVE_BEGIN__\n'
tar -C "$run_root" -czf - . | base64
printf '__KERNELGEN_ARCHIVE_END__\n'
REMOTE
  } | multi_device_remote_bash \
    "$mode" "$address" "$ssh_port" "$container" "$jump_login" |
    awk '
      /^__KERNELGEN_ARCHIVE_BEGIN__$/ { copying=1; next }
      /^__KERNELGEN_ARCHIVE_END__$/ { copying=0; found_end=1; exit }
      copying { print }
      END { if (!found_end) exit 4 }
    ' |
    base64 --decode >"$partial"

  tar -tzf "$partial" >/dev/null
  mv "$partial" "$archive"

  if ((extract)); then
    extract_root="$output/$name/$run_name"
    [[ ! -e "$extract_root" ]] || {
      printf 'refusing to overwrite extracted directory: %s\n' \
        "$extract_root" >&2
      exit 2
    }
    mkdir -p "$extract_root"
    tar -xzf "$archive" -C "$extract_root"
  fi
done <<< "$records"
