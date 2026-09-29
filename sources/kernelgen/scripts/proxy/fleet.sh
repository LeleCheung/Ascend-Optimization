#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
repo_root=$(cd -- "$script_dir/../.." >/dev/null 2>&1 && pwd)
export PYTHONPATH="$repo_root/..${PYTHONPATH:+:$PYTHONPATH}"

exec python3 -m kernelgen.service.fleet_daemon "$@"
