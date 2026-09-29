---
schema_version: '1.0'
id: kg:diagnostic:triton-ascend-profile-bottleneck-routing
kind: diagnostic
title: Route Triton Ascend optimization from measured pipeline bottlenecks
summary: Use measured compute, memory, scalar and pipeline behavior to identify the
  dominant bottleneck before selecting an optimization.
claim_key: diagnostic.triton_ascend.profile_bottleneck_routing
domains:
- diagnosis
- optimization
- profiling
status: stable
verified:
- by: kernelgen/publisher-v1
  at: '2026-07-27T05:54:19.817001Z'
sources:
- resource: source:triton-ascend.a60006ac63f3
  locator: docs/en/debug_guide/profiling.md#Locating Bottlenecks
  title: Route Triton Ascend optimization from measured pipeline bottlenecks
scope:
  target:
    level: backend
    backend: ascend
    software:
      language: triton
      compiler: triton-ascend
  numerics:
    exact: false
retrieval:
  phases:
  - post_profile
  - plateau
  tasks:
  - diagnosis
  - next_experiment
  symptoms:
  - pipeline_stall
  - scalar_fallback
  - memory_bound
  - compute_bound
  techniques:
  - profile_guided_optimization
  keywords:
  - profiling
  - bottleneck
  - pipeline
  - scalar
  - memory
  - compute
evidence_state: source_supported
managed:
  content_hash: sha256:9e4a037ef7c13cd73ac1d8079f343853f9664391420601054ecee815dad9c598
  created_by: kernelgen/publisher-v1
  created_at: '2026-07-27T05:54:19.817001Z'
  updated_at: '2026-07-27T05:54:19.817001Z'
---

# Claim

Use measured compute, memory, scalar and pipeline behavior to identify the dominant bottleneck before selecting an optimization.

# Source material

### Locating Bottlenecks

After the performance data is obtained, processes that deviate significantly from theoretical values or consume excessive time are identified as "bottlenecks." The following describes how to find bottlenecks and corresponding optimization directions based on performance data.

- Method 1: Use board profiling to analyze the pipeline.
View the **op_summary_\*.csv** file parsed by board profiling to analyze the pipeline. Note that "\*" indicates the timestamp.
![analyse_data_op_summary](../figures/performance_analysis_analyse_data_op_summary.png)

    In ideal cases, the utilization rate of each pipeline should be 100%. Any pipeline falling short of this target represents room for improvement. The preceding figure shows the data obtained from an AI processor. In the first scenario of the Vector operator _layer_norm_fwd_fused, the Vector pipeline utilization **aiv_vec_ratio** is less than 10%, indicating that the computing power is not fully utilized. The Scalar pipeline utilization **aiv_scalar_ratio** is about 60%, indicating that Scalar is the longest pipeline. \
    When Scalar is the longest pipeline, analyze whether complex operations are performed on scalar values in the operator source code. The SIMD microarchitecture of Ascend is more suitable for multi-data parallel computing. Another possibility is that the Triton software stack degrades vector computing to scalar computing because some instructions do not support specific data types on the hardware. Optimization should involve both pipeline and scalar optimization methods. For details, see method 3 to view the simulation pipeline diagram and method 4 to view the code hotspots for further analysis. \
    For more general cases such as MTE2 data transfer and actual scenarios: The shapes of the three input matrices are (128,128), (128,1), and (128,1), respectively, and the data type is float16. The current algorithm uses the two-pass method. Therefore, X is moved in for three times, and W and B are moved in for one time. The total amount of data to be transferred can be calculated accordingly. The theoretical value calculated based on the method described in the [Theoretical Parameters](#theoretical-parameters) section is sizeof(float16) *(128* 128 * 3 + 128 + 128)/1.8 TB/s ≈ 0.1991 μs (calculated based on 1 TB = 10<sup>12</sup> Byte), which is greatly different from the actual performance data aiv_mte2_time. Analysis shows the total input size is smaller than the Unified Buffer (UB) capacity (192 KB for the A2 model). Therefore, if the MTE2 time is excessive, the basic block obtained through tiling computation may be too small, triggering redundant transfer instructions. In this case, pipeline optimization and tiling optimization are required, you can refer to method 3 to view the simulation pipeline diagram and analyze each pipeline for further analysis.

- Method 2: Use board profiling to analyze the tiling.
The AI processor used in the previous example has 48 vector cores. The _layer_norm_fwd_fused operator is a pure vector operator. However, in some scenarios, too many blocks (Block Dim > 48) are delivered, causing excessive host scheduling overhead. In this case, the next step is to optimize the tiling.

- Method 3: Use the simulation pipeline diagram to analyze the pipeline.
![analyse_data_waveform](../figures/performance_analysis_analyse_data_waveform.png) \
    The preceding figure shows the data obtained from an AI processor simulator. It can be seen that the SCALAR and FLOWCTRL instructions of the Vector core are saturated. You can analyze the operator logic to check whether there are too many scalar computations and unsupported vectorization operations. The next step is to optimize scalar computation. On the other hand, the related pipelines (such as MTE2 and VECTOR of veccore0) of the Vector core are regularly interrupted, that is, there are a large number of blank segments without operations. You can analyze the operator logic to check whether the stream interruption is caused by small basic block splitting. The main optimization direction is pipeline optimization. In addition, the vector pipeline utilization is further improved using tiling optimization and memory optimization.

- Method 4: Analyze the code hotspot.
![analyse_data_code_mapping](../figures/performance_analysis_analyse_data_code_mapping.png) \
    The preceding figure shows the data obtained from an AI processor simulator. The load interface on the left corresponds to a group of assembly instructions on the right (only instructions related to code lines are displayed and sorted in descending order by cycle count). The high proportion of scalar instructions is inconsistent with the scenario where the MTE proportion should be high when load is used as the memory access interface. Therefore, the main optimization direction is scalar calculation.

# Applicability

The structured scope on this Concept is normative.
