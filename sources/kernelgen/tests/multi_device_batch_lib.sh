#!/usr/bin/env bash

multi_device_script_dir=$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)

: "${MULTI_DEVICE_INVENTORY:=$multi_device_script_dir/hosts.md}"
: "${MULTI_DEVICE_IDENTITY:=${HOME}/.ssh/id_ed25519}"
: "${MULTI_DEVICE_BASTION:=bastion.aiops.baai.ac.cn}"

# Markdown inventories have exactly one fenced machine-readable section.
# Explicit --inventory plain-text files remain usable by existing callers.
multi_device_inventory_records() {
  if [[ "$MULTI_DEVICE_INVENTORY" != *.md ]]; then
    cat -- "$MULTI_DEVICE_INVENTORY"
    return
  fi
  awk '
    $0 == "```kg-hosts" { blocks++; inside = 1; next }
    inside && $0 == "```" { inside = 0; next }
    inside { print }
    END { if (blocks != 1 || inside) exit 2 }
  ' "$MULTI_DEVICE_INVENTORY"
}

multi_device_validate_record() {
  local name=$1
  local mode=$2
  local address=$3
  local ssh_port=$4
  local container=$5
  local deploy_base=$6
  local server_port=$7
  local jump_login=$8

  [[ "$name" =~ ^[a-z0-9_-]+$ ]] || return 1
  [[ "$mode" == jump || "$mode" == direct ]] || return 1
  [[ "$address" =~ ^[A-Za-z0-9._:-]+$ ]] || return 1
  [[ "$ssh_port" =~ ^[0-9]+$ ]] || return 1
  [[ "$server_port" =~ ^[0-9]+$ ]] || return 1
  [[ "$deploy_base" == /* ]] || return 1
  if [[ "$mode" == jump ]]; then
    [[ "$container" =~ ^[A-Za-z0-9_.-]+$ ]] || return 1
    [[ "$jump_login" =~ ^[A-Za-z0-9@#._:-]+$ ]] || return 1
  else
    [[ "$container" == "-" ]] || return 1
    [[ "$jump_login" == "-" ]] || return 1
  fi
}

multi_device_selected() {
  local candidate=$1
  shift
  local selected
  if (($# == 0)); then
    return 0
  fi
  for selected in "$@"; do
    if [[ "$selected" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

multi_device_remote_bash() {
  local mode=$1
  local address=$2
  local ssh_port=$3
  local container=$4
  local jump_login=$5

  if [[ "$mode" == jump ]]; then
    local remote_command="bash -s"
    if [[ "$container" != "-" ]]; then
      remote_command="sudo docker exec -i $container bash -s"
    fi
    ssh -T \
        -o BatchMode=yes \
        -o ConnectTimeout=20 \
        -p "$ssh_port" \
        -i "$MULTI_DEVICE_IDENTITY" \
        -l "$jump_login" \
        "$MULTI_DEVICE_BASTION" \
        "$remote_command"
  else
    ssh -T \
      -o BatchMode=yes \
      -o ConnectTimeout=20 \
      -p "$ssh_port" \
      -i "$MULTI_DEVICE_IDENTITY" \
      "root@$address" \
      "bash -s"
  fi
}
