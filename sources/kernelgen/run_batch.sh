#!/usr/bin/env bash

source env.sh

export PYTHONPATH=/data/jiabei/KernelGen-Next/jiabei/kernelgen_server:/data/jiabei/KernelGen-Next
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

definitions=(
  ks_fused_moe_e8_k2_h128_i64
  # ks_grouped_topk_softmax_e8_k8_g8_tg4
  # ks_hc_split_sinkhorn_hc4_iter20
  # ks_mhc_post
  # ks_mm_encoder_attention_h8_d64_kv8
  # ks_causal_sdpa_t83_h8_d64
  # ks_centre_random_augmentation_s4
  # ks_head_compute_mix_bwd_m4
  # ks_music_flamingo_rope_d64_s256
  # ks_splade_pool_max
)
target_hardware="Ascend910B3"
knowledge_reviewer_mode="${KERNELGEN_KNOWLEDGE_REVIEWER_MODE:-off}"
workspace="/data/jiabei/KernelGen-Next/runs/kernelgen_batch_"$target_hardware"_$(date +%Y%m%d_%H%M%S)"
# workspace="/data/jiabei/KernelGen-Next/runs/kernelgen_batch_20260812_120223"

python3 -u examples/kernel_gen/run_campaign.py \
  --definitions "${definitions[@]}" \
  --workspace-root "$workspace" \
  --eval-server http://127.0.0.1:19405 \
  --target-hardware "$target_hardware" \
  --catalog-name kernelswift \
  --implementation-language triton \
  --knowledge-catalog-path /data/jiabei/kernelgen-kb-kernelswift-competition \
  --knowledge-reviewer-mode "$knowledge_reviewer_mode" \
  --max-operators 5 \
  --agents-per-operator 2 \
  --n-epoch 2 \
  --early-stop-rounds 5 \
  --min-rounds 2 \
  --max-round 40 \
  --eval-tolerance-mode fixed \
  --eval-atol 0.01 \
  --eval-rtol 0.01 \
  --eval-required-matched-ratio 1.0 \
  --timeout 3600
