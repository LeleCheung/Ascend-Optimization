source env.sh

export PYTHONPATH=/data/jiabei/KernelGen-Next/jiabei/kernelgen_server:/data/jiabei/KernelGen-Next
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

definition=ks_fused_moe_e8_k2_h128_i64
workspace="/data/jiabei/KernelGen-Next/runs/${definition}_$(date +%Y%m%d_%H%M%S)"

python3 -u examples/kernel_gen/run_example.py \
  --definition "$definition" \
  --workspace "$workspace" \
  --eval-server http://127.0.0.1:19106 \
  --target-hardware Ascend910B-1 \
  --catalog-name kernelswift \
  --implementation-language triton \
  --knowledge-mode read_write_v1 \
  --knowledge-catalog-path /data/jiabei/kernelgen-kb-test-kernelswift \
  --start-mode fresh \
  --n-parallel 2 \
  --n-epoch 2 \
  --early-stop-rounds 5 \
  --min-rounds 2 \
  --max-round 40 \
  --eval-tolerance-mode fixed \
  --eval-atol 0.01 \
  --eval-rtol 0.01 \
  --eval-required-matched-ratio 1.0 \
  --timeout 3600
