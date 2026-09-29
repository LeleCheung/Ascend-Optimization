# Historical retrieval replay: baseline-c104631

- Cohort: `kernelswift-published-workspaces-v1`
- Published workspaces: 49 (46 with retrieval logs, 3 without)
- Successful queries: 301 (158 Concept + 143 Source)
- Queries followed by a successful detail read: 75 (24.9%)

## Interpretation

Historical detail reads are a change-audit proxy, not relevance labels: they reflect the legacy ranker, legacy catalog snapshots, and what an Agent chose to inspect. Candidate acceptance is determined by required golden cases and strict Scope checks; detail-retention gains and losses identify cases for semantic review.

## Replay metrics

| Operation | Queries | Errors | Empty Top-4 | Historical Top-1 retained | Exact Top-4 | Mean Jaccard | Detail-query hit rate | Detail-ref recall | Incompatible direct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `query_knowledge` | 158 | 0 | 10 | 0 | 10 | 14.4% | 37.7% | 21.4% | 0 |
| `query_sources` | 143 | 0 | 0 | 116 | 69 | 75.2% | 95.5% | 95.7% | 0 |

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
