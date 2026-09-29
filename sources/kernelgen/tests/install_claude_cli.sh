#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  install_claude_cli.sh \
    --base DIR \
    [--version VERSION] \
    [--node-version VERSION]

Install a pinned Claude Code package into:
  DIR/runtime/claude/node_modules/

When the target machine does not provide Node.js >= 22 and npm, install a
verified official Node.js release into DIR/runtime/. The installation is local
to DIR and does not modify global Node.js or npm packages.
EOF
}

base=""
version="2.1.220"
node_version="22.23.1"

while (($#)); do
  case "$1" in
    --base)
      base=$2
      shift 2
      ;;
    --version)
      version=$2
      shift 2
      ;;
    --node-version)
      node_version=$2
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

if [[ -z "$base" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+-][A-Za-z0-9._-]+)?$ ]]; then
  printf 'version must be an exact npm package version, got: %s\n' "$version" >&2
  exit 2
fi
if [[ ! "$node_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  printf 'node-version must be an exact Node.js version, got: %s\n' \
    "$node_version" >&2
  exit 2
fi

runtime_root="$base/runtime"
mkdir -p "$runtime_root"

node_major=0
if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
  node_major=$(node -p 'Number(process.versions.node.split(".")[0])')
fi

if [[ ! "$node_major" =~ ^[0-9]+$ ]] || ((node_major < 22)); then
  case "$(uname -m)" in
    x86_64)
      node_arch="x64"
      ;;
    aarch64|arm64)
      node_arch="arm64"
      ;;
    *)
      printf 'unsupported CPU architecture for Node.js: %s\n' "$(uname -m)" >&2
      exit 1
      ;;
  esac

  node_archive="node-v${node_version}-linux-${node_arch}.tar.xz"
  node_dir="$runtime_root/node-v${node_version}-linux-${node_arch}"
  node_archive_path="$runtime_root/$node_archive"
  checksum_path="$runtime_root/node-v${node_version}-SHASUMS256.txt"
  node_base_url="https://nodejs.org/dist/v${node_version}"

  if [[ ! -x "$node_dir/bin/node" || ! -x "$node_dir/bin/npm" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
      printf 'curl is required to install the isolated Node.js runtime\n' >&2
      exit 1
    fi
    curl --fail --location --silent --show-error \
      --output "$node_archive_path" \
      "$node_base_url/$node_archive"
    curl --fail --location --silent --show-error \
      --output "$checksum_path" \
      "$node_base_url/SHASUMS256.txt"

    expected_sha=$(
      awk -v archive="$node_archive" '$2 == archive {print $1}' "$checksum_path"
    )
    actual_sha=$(sha256sum "$node_archive_path" | awk '{print $1}')
    if [[ -z "$expected_sha" || "$actual_sha" != "$expected_sha" ]]; then
      printf 'Node.js archive checksum mismatch: %s\n' "$node_archive" >&2
      exit 1
    fi

    tar -xJf "$node_archive_path" -C "$runtime_root"
  fi

  export PATH="$node_dir/bin:$PATH"
  ln -sfn "$node_dir/bin" "$runtime_root/node-bin"
fi

runtime_dir="$base/runtime/claude"
mkdir -p "$runtime_dir"
if [[ ! -f "$runtime_dir/package.json" ]]; then
  npm --prefix "$runtime_dir" init --yes >/dev/null
fi

claude_bin="$runtime_dir/node_modules/.bin/claude"
installed_version=""
if [[ -f "$runtime_dir/package-lock.json" && -x "$claude_bin" ]]; then
  if current_output=$("$claude_bin" --version 2>/dev/null); then
    installed_version=${current_output%% *}
  fi
fi

if [[ "$installed_version" == "$version" ]]; then
  printf 'npm_install=already_satisfied\n'
else
  npm --prefix "$runtime_dir" install \
    --save-exact \
    --no-audit \
    --no-fund \
    --fetch-retries 5 \
    --fetch-retry-factor 2 \
    --fetch-retry-mintimeout 20000 \
    --fetch-retry-maxtimeout 120000 \
    --fetch-timeout 600000 \
    "@anthropic-ai/claude-code@$version"
fi

if [[ ! -x "$claude_bin" ]]; then
  printf 'Claude CLI was not installed at expected path: %s\n' "$claude_bin" >&2
  exit 1
fi

actual_version=$("$claude_bin" --version)
printf 'node=%s\n' "$(node --version)"
printf 'npm=%s\n' "$(npm --version)"
printf 'claude=%s\n' "$claude_bin"
printf 'version=%s\n' "$actual_version"
sha256sum "$(readlink -f "$claude_bin")"
