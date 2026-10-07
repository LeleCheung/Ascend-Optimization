#!/usr/bin/env bash
set -euo pipefail

container=${CONTAINER:-tle_yy}
root=/data/hanle/ascend-optimization/FlagGems
rev=${1:?usage: verify-phase1-case-api-910b.sh REVISION}

docker exec "$container" git -C "$root" fetch origin main
docker exec "$container" git -C "$root" checkout --detach "$rev"
docker exec "$container" git -C "$root" rev-parse HEAD

python=/usr/local/python3.11.15/bin/python3.11
for spec in \
  "test_mse_loss_backward.py:mse_loss_backward" \
  "test_prelu.py:prelu" \
  "test_t_copy.py:t_copy" \
  "test_batch_norm_backward.py:batch_norm_backward" \
  "test_smooth_l1_loss.py:smooth_l1_loss_backward"; do
  file=${spec%%:*}
  mark=${spec##*:}
  docker exec "$container" env PYTHONPATH="$root/src" "$python" -m pytest \
    "$root/benchmark/$file" -k "$mark" --list-cases -q
done
