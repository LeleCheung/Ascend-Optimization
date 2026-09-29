#!/bin/bash
# Sync kernelgen code to the 910B-6 docker container.
set -euo pipefail

LOCAL_ROOT="/data/akg_kernel_bench_lite/kernelgen"
REMOTE_ROOT="/data/xuyao/kernelgen"
SSH_ARGS="-T -p 2224 -i $HOME/.ssh/id_ed25519 -l xuyao@secure@10.0.0.6 bastion.aiops.baai.ac.cn"

remote_exec() {
    ssh $SSH_ARGS "sudo docker exec -i xuyao_sglang bash -c '$1'" 2>/dev/null
}

cd "$LOCAL_ROOT"

echo "📦 Packing kernelgen repo..."
# Include tracked files, non-ignored new files, and tmp/ dummy data needed by e2e.
(
    git ls-files
    git ls-files --others --exclude-standard
    find tmp -type f 2>/dev/null
) | sort -u | while IFS= read -r f; do
    [ -f "$f" ] && printf '%s\n' "$f"
done | tar czf /tmp/_sync_kernelgen.tar.gz -T -
base64 /tmp/_sync_kernelgen.tar.gz | remote_exec "base64 -d > /tmp/_sync_kernelgen.tar.gz && mkdir -p $REMOTE_ROOT && tar xzf /tmp/_sync_kernelgen.tar.gz -C $REMOTE_ROOT && rm /tmp/_sync_kernelgen.tar.gz"
rm -f /tmp/_sync_kernelgen.tar.gz
echo "✅ Synced → $REMOTE_ROOT"
