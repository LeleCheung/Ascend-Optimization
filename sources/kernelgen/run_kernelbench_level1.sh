#!/usr/bin/env bash

source env.sh

export PYTHONPATH=/data/jiabei/KernelGen-Next/jiabei/kernelgen_server:/data/jiabei/KernelGen-Next
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

definitions_dir=/data/jiabei/KernelGen-Next/jiabei/kernelgen_server/data/kernelbench/ops/level1
mapfile -t definitions < <(
  find "$definitions_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
)

target_hardware="A100-SXM4"
workspace="/data/jiabei/KernelGen-Next/runs/kernelbench_level1_$(date +%Y%m%d_%H%M%S)"
# workspace="/data/jiabei/KernelGen-Next/runs/kernelbench_level1_20260814_000000"

python3 -u examples/kernel_gen/run_campaign.py \
  --definitions "${definitions[@]}" \
  --workspace-root "$workspace" \
  --eval-server http://127.0.0.1:18081 \
  --target-hardware "$target_hardware" \
  --catalog-name kernelbench \
  --implementation-language triton \
  --knowledge-catalog-path /data/jiabei/kernelgen-kb-source-kernelbench \
  --max-operators 10 \
  --agents-per-operator 2 \
  --start-mode fresh \
  --n-epoch 1 \
  --early-stop-rounds 5 \
  --min-rounds 2 \
  --max-round 40 \
  --eval-tolerance-mode fixed \
  --eval-atol 0.0001 \
  --eval-rtol 0.0001 \
  --eval-required-matched-ratio 1.0 \
  --no-profile \
  --timeout 3600
