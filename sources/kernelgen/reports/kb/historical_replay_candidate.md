# Historical retrieval replay: candidate-bm25-rrf

- Cohort: `kernelswift-published-workspaces-v1`
- Published workspaces: 49 (46 with retrieval logs, 3 without)
- Successful queries: 301 (158 Concept + 143 Source)
- Queries followed by a successful detail read: 75 (24.9%)

## Interpretation

Historical detail reads are a change-audit proxy, not relevance labels: they reflect the legacy ranker, legacy catalog snapshots, and what an Agent chose to inspect. Candidate acceptance is determined by required golden cases and strict Scope checks; detail-retention gains and losses identify cases for semantic review.

## Replay metrics

| Operation | Queries | Errors | Empty Top-4 | Historical Top-1 retained | Exact Top-4 | Mean Jaccard | Detail-query hit rate | Detail-ref recall | Incompatible direct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `query_knowledge` | 158 | 0 | 10 | 15 | 10 | 19.7% | 50.9% | 31.6% | 0 |
| `query_sources` | 143 | 0 | 0 | 1 | 0 | 2.5% | 27.3% | 26.1% | 0 |

## Coverage

### Query language

| Language | Queries |
|---|---:|
| english | 299 |
| mixed | 2 |

### Definition

| Definition | Queries |
|---|---:|
| `abl_t2_add_rmsnorm_quant` | 14 |
| `kernelbench_l1_p001_square_matrix_multiplication` | 2 |
| `kernelbench_l1_p002_standard_matrix_multiplication` | 7 |
| `kernelbench_l1_p003_batched_matrix_multiplication` | 2 |
| `kernelbench_l1_p005_matrix_scalar_multiplication` | 2 |
| `kernelbench_l1_p006_matmul_with_large_k_dimension` | 4 |
| `kernelbench_l1_p007_matmul_with_small_k_dimension` | 4 |
| `kernelbench_l1_p008_matmul_with_irregular_shapes` | 5 |
| `kernelbench_l1_p010_3d_tensor_matrix_multiplication` | 3 |
| `kernelbench_l1_p011_4d_tensor_matrix_multiplication` | 1 |
| `kernelbench_l1_p019_relu` | 2 |
| `ks_fused_moe_e8_k2_h128_i64` | 74 |
| `ks_grouped_topk_softmax_e8_k8_g8_tg4` | 44 |
| `ks_hc_split_sinkhorn_hc4_iter20` | 63 |
| `ks_head_compute_mix_bwd_m4` | 51 |
| `ks_mhc_post` | 23 |

## Comparison to baseline-c104631

| Operation | Top-1 changes | Top-4 changes | Mean overlap | Detail gains | Detail losses | Detail-rank better | Detail-rank worse | Strict incompatible baseline | Strict incompatible candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `query_knowledge` | 76 | 141 | 50.1% | 8 | 1 | 23 | 3 | 85 | 0 |
| `query_sources` | 141 | 142 | 3.4% | 0 | 15 | 0 | 21 | 0 | 0 |

### Detail-read retention changes

| Query | Operation | Definition | Change |
|---|---|---|---|
| `query:052a34dd9047418a82d9f2b0c344ba76` | `query_knowledge` | `ks_fused_moe_e8_k2_h128_i64` | gain |
| `query:25a4260c96604f84bf011aba5ab0bd88` | `query_knowledge` | `ks_hc_split_sinkhorn_hc4_iter20` | loss |
| `query:40daf4a1221048f39d05870f450d07ad` | `query_knowledge` | `ks_head_compute_mix_bwd_m4` | gain |
| `query:4bce2a89356d446d9d39ea8be7dfb066` | `query_knowledge` | `ks_mhc_post` | gain |
| `query:9852711b2ddb49ea91dab329bbbe7328` | `query_knowledge` | `ks_fused_moe_e8_k2_h128_i64` | gain |
| `query:abcfe2ed165144ceb824fb2bccc43b23` | `query_knowledge` | `ks_grouped_topk_softmax_e8_k8_g8_tg4` | gain |
| `query:b203d6cde0e54abb91acab7011efc90d` | `query_knowledge` | `ks_head_compute_mix_bwd_m4` | gain |
| `query:c036a3ab75994bb38e079b91045da402` | `query_knowledge` | `ks_hc_split_sinkhorn_hc4_iter20` | gain |
| `query:d0854cac563d4af1a3e7d9bd1278b6f0` | `query_knowledge` | `ks_fused_moe_e8_k2_h128_i64` | gain |
| `source-query:004df4ce991f4db6bce6133716014e02` | `query_sources` | `ks_fused_moe_e8_k2_h128_i64` | loss |
| `source-query:2f22d111bcd14022974b68562f8d6ca8` | `query_sources` | `ks_head_compute_mix_bwd_m4` | loss |
| `source-query:3743668b82eb42c0a7d9db57bfe49d47` | `query_sources` | `ks_hc_split_sinkhorn_hc4_iter20` | loss |
| `source-query:4bc2edad32d5495fbf7ac1469590ab2c` | `query_sources` | `ks_fused_moe_e8_k2_h128_i64` | loss |
| `source-query:5d0d212f353b4e26aece27866918dc8b` | `query_sources` | `ks_fused_moe_e8_k2_h128_i64` | loss |
| `source-query:68e2a33938044242857df509ca170381` | `query_sources` | `ks_mhc_post` | loss |
| `source-query:86582ff813e0476db6f3aa5d03887f0b` | `query_sources` | `ks_fused_moe_e8_k2_h128_i64` | loss |
| `source-query:89f2f3b3679c4851b31cbcc339411bfd` | `query_sources` | `ks_grouped_topk_softmax_e8_k8_g8_tg4` | loss |
| `source-query:9cd58aaa31554138bb5e51d3151c782f` | `query_sources` | `ks_head_compute_mix_bwd_m4` | loss |
| `source-query:a8de6846df9d4f9d9e203c458cfa90f7` | `query_sources` | `ks_hc_split_sinkhorn_hc4_iter20` | loss |
| `source-query:a8f3df2e46474926ba99521f38286d8a` | `query_sources` | `ks_mhc_post` | loss |
| `source-query:acdb49c649cb49fdbe4e41cc723cea44` | `query_sources` | `ks_fused_moe_e8_k2_h128_i64` | loss |
| `source-query:bc0cb7095aeb410db32dbfe3ea8c894f` | `query_sources` | `ks_hc_split_sinkhorn_hc4_iter20` | loss |
| `source-query:c468dea9d59949379101b454b93f0ea6` | `query_sources` | `ks_fused_moe_e8_k2_h128_i64` | loss |
| `source-query:f1be1c4dde3e450cb3eba13b105cdddb` | `query_sources` | `ks_grouped_topk_softmax_e8_k8_g8_tg4` | loss |
