你是Ascend910B性能分析角色。不要调用工具。当前目标是实际加速matmul_bias_activation，保留多输出小tile，避免上一轮单输出变慢。请根据已有Roofline、源码IR事实与本轮真实搜索记录，选择最多两项尚未测的追加配置。只输出JSON：config_ids（不重复数组）、hypothesis（中文）、predictions（指标与预期方向数组）、limitations（数组）。config_ids只能来自["dual-n-group4", "dual-n-persistent20", "dual-n-stage1"]。不可把limiting_side=compute直接解释为实际计算受限，不可把流水活跃比例相加或等同整卡利用率。GM、L2事件模型、逻辑字节与算法量区分。既有UB超限与单输出回归不得重新包装为收益。本轮候选仅半精度快路径，FP32回父版。依据大形状交错真实设备计时选版，一次完整KGS裁决。

Roofline冻结证据：
{
  "schema_version": "hgkda-roofline-v1",
  "source": {
    "path": "E:\\0_XSAI\\00_KernelAgent\\kernelgen\\Ascend-Optimization\\project\\Ascend-Optimization-v1\\operators\\campaigns\\roofline-20261027\\analysis-final\\diagnosis.json",
    "sha256": "99cf2d060b1bfdc2cc7beec5e11b47267d8031ccf0aee3c3f3aa01be08486ee6"
  },
  "cases": [
    {
      "id": "matrix-profile/parent-float16-1024",
      "spec": {
        "side": "parent",
        "shape": [
          1024,
          1024,
          1024
        ],
        "dtype": "float16",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/device-night-20261026-v11/seeds/matmul_bias_activation.py",
        "source_sha256": "9d2cd2f13c6bd5c43cbbf7ce823e8175664948bd1752125b53982dffef834061"
      },
      "kernel": "mba_dual_dispatch_v8_kernel_mix_aic",
      "duration_us": 36.280727,
      "algorithmic_flops": 2147483648,
      "paths": [
        {
          "component": "GM/L2",
          "path": "GM Read + Write",
          "official_bytes": 3146879.9835887207,
          "official_flops": 2154917878.3712726,
          "hardware_over_algorithmic": 1.003461833284829,
          "algorithmic_ai": 682.4167617447542,
          "algorithmic_tflops": 59.19075568689679,
          "reference_compute_tflops": 248.59945314939048,
          "reference_bandwidth_gbps": 1147.601937244727,
          "reference_bound_tflops": 248.59945314939048,
          "reference_achievement": 0.23809688612358848,
          "limiting_side": "compute",
          "reference_kind": "经验观察包络；不是已证明物理上界"
        },
        {
          "component": "GM/L2",
          "path": "L2 Read + Write",
          "official_bytes": 31514622.688309554,
          "official_flops": 2154917878.3712726,
          "hardware_over_algorithmic": 1.003461833284829,
          "algorithmic_ai": 68.14245149749534,
          "algorithmic_tflops": 59.19075568689679,
          "reference_compute_tflops": 248.59945314939048,
          "reference_bandwidth_gbps": 4000.0,
          "reference_bound_tflops": 248.59945314939048,
          "reference_achievement": 0.23809688612358848,
          "limiting_side": "compute",
          "reference_kind": "官方L2带宽模型＋同精度经验有效计算参考"
        }
      ],
      "pipeline": {
        "aic_cube_ratio": {
          "mean": 0.14262046875,
          "minimum": 0.141195,
          "maximum": 0.143367,
          "count": 32
        },
        "aic_mte1_ratio": {
          "mean": 0.13657934375,
          "minimum": 0.132626,
          "maximum": 0.139747,
          "count": 32
        },
        "aic_mte2_ratio": {
          "mean": 0.27783884375,
          "minimum": 0.173165,
          "maximum": 0.347306,
          "count": 32
        },
        "aic_scalar_ratio": {
          "mean": 0.12885312499999999,
          "minimum": 0.073502,
          "maximum": 0.216397,
          "count": 32
        },
        "aic_scalar_mte1_stall_time(us)": {
          "mean": 7.2822538125,
          "minimum": 5.014545,
          "maximum": 9.870303,
          "count": 32
        },
        "aiv_vec_ratio": {
          "mean": 0.0583,
          "minimum": 0.0583,
          "maximum": 0.0583,
          "count": 64
        },
        "aiv_scalar_ratio": {
          "mean": 0.052674671875,
          "minimum": 0.009781,
          "maximum": 0.12436,
          "count": 64
        }
      },
      "issues": [
        "官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界"
      ]
    },
    {
      "id": "matrix-profile/parent-float16-2048",
      "spec": {
        "side": "parent",
        "shape": [
          2048,
          2048,
          2048
        ],
        "dtype": "float16",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/device-night-20261026-v11/seeds/matmul_bias_activation.py",
        "source_sha256": "9d2cd2f13c6bd5c43cbbf7ce823e8175664948bd1752125b53982dffef834061"
      },
      "kernel": "mba_dual_dispatch_v8_kernel_mix_aic",
      "duration_us": 149.062988,
      "algorithmic_flops": 17179869184,
      "paths": [
        {
          "component": "GM/L2",
          "path": "GM Read + Write",
          "official_bytes": 12585088.251299184,
          "official_flops": 17209605824.782227,
          "hardware_over_algorithmic": 1.0017309003033574,
          "algorithmic_ai": 1365.097235788274,
          "algorithmic_tflops": 115.2524138587642,
          "reference_compute_tflops": 248.59945314939048,
          "reference_bandwidth_gbps": 1147.601937244727,
          "reference_bound_tflops": 248.59945314939048,
          "reference_achievement": 0.46360686799059747,
          "limiting_side": "compute",
          "reference_kind": "经验观察包络；不是已证明物理上界"
        },
        {
          "component": "GM/L2",
          "path": "L2 Read + Write",
          "official_bytes": 226721786.8822705,
          "official_flops": 17209605824.782227,
          "hardware_over_algorithmic": 1.0017309003033574,
          "algorithmic_ai": 75.77511372085722,
          "algorithmic_tflops": 115.2524138587642,
          "reference_compute_tflops": 248.59945314939048,
          "reference_bandwidth_gbps": 4000.0,
          "reference_bound_tflops": 248.59945314939048,
          "reference_achievement": 0.46360686799059747,
          "limiting_side": "compute",
          "reference_kind": "官方L2带宽模型＋同精度经验有效计算参考"
        }
      ],
      "pipeline": {
        "aic_cube_ratio": {
          "mean": 0.0690157734375,
          "minimum": 0.068326,
          "maximum": 0.069546,
          "count": 128
        },
        "aic_mte1_ratio": {
          "mean": 0.0664291875,
          "minimum": 0.063262,
          "maximum": 0.068098,
          "count": 128
        },
        "aic_mte2_ratio": {
          "mean": 0.10148359375,
          "minimum": 0.067993,
          "maximum": 0.163181,
          "count": 128
        },
        "aic_scalar_ratio": {
          "mean": 0.027365984375,
          "minimum": 0.018646,
          "maximum": 0.060757,
          "count": 128
        },
        "aic_scalar_mte1_stall_time(us)": {
          "mean": 13.13277459375,
          "minimum": 9.155151,
          "maximum": 21.735758,
          "count": 128
        },
        "aiv_vec_ratio": {
          "mean": 0.014086,
          "minimum": 0.014086,
          "maximum": 0.014086,
          "count": 256
        },
        "aiv_scalar_ratio": {
          "mean": 0.0077210390625,
          "minimum": 0.002034,
          "maximum": 0.035993,
          "count": 256
        }
      },
      "issues": [
        "官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界"
      ]
    },
    {
      "id": "matrix-profile/parent-float32-1024",
      "spec": {
        "side": "parent",
        "shape": [
          1024,
          1024,
          1024
        ],
        "dtype": "float32",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/device-night-20261026-v11/seeds/matmul_bias_activation.py",
        "source_sha256": "9d2cd2f13c6bd5c43cbbf7ce823e8175664948bd1752125b53982dffef834061"
      },
      "kernel": "mba_fp32_dual_v9_kernel_mix_aic",
      "duration_us": 58.101162,
      "algorithmic_flops": 2147483648,
      "paths": [
        {
          "component": "GM/L2",
          "path": "GM Read + Write",
          "official_bytes": 8392832.76134448,
          "official_flops": 8597342490.896303,
          "hardware_over_algorithmic": 4.003449571736298,
          "algorithmic_ai": 255.8711354157838,
          "algorithmic_tflops": 36.96111358323608,
          "reference_compute_tflops": 66.27840976514481,
          "reference_bandwidth_gbps": 1147.601937244727,
          "reference_bound_tflops": 66.27840976514481,
          "reference_achievement": 0.557664459877756,
          "limiting_side": "compute",
          "reference_kind": "经验观察包络；不是已证明物理上界"
        },
        {
          "component": "GM/L2",
          "path": "L2 Read + Write",
          "official_bytes": 54616064.9228604,
          "official_flops": 8597342490.896303,
          "hardware_over_algorithmic": 4.003449571736298,
          "algorithmic_ai": 39.31963335390605,
          "algorithmic_tflops": 36.96111358323608,
          "reference_compute_tflops": 66.27840976514481,
          "reference_bandwidth_gbps": 4000.0,
          "reference_bound_tflops": 66.27840976514481,
          "reference_achievement": 0.557664459877756,
          "limiting_side": "compute",
          "reference_kind": "官方L2带宽模型＋同精度经验有效计算参考"
        }
      ],
      "pipeline": {
        "aic_cube_ratio": {
          "mean": 0.342853,
          "minimum": 0.342853,
          "maximum": 0.342853,
          "count": 32
        },
        "aic_mte1_ratio": {
          "mean": 0.15985609375,
          "minimum": 0.15488,
          "maximum": 0.163145,
          "count": 32
        },
        "aic_mte2_ratio": {
          "mean": 0.2693880625,
          "minimum": 0.15837,
          "maximum": 0.339143,
          "count": 32
        },
        "aic_scalar_ratio": {
          "mean": 0.09599003125,
          "minimum": 0.074504,
          "maximum": 0.143229,
          "count": 32
        },
        "aic_scalar_mte1_stall_time(us)": {
          "mean": 17.84937503125,
          "minimum": 15.992121,
          "maximum": 19.619394,
          "count": 32
        },
        "aiv_vec_ratio": {
          "mean": 0.032905,
          "minimum": 0.032905,
          "maximum": 0.032905,
          "count": 64
        },
        "aiv_scalar_ratio": {
          "mean": 0.028422984375,
          "minimum": 0.005549,
          "maximum": 0.069039,
          "count": 64
        }
      },
      "issues": [
        "官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界"
      ]
    },
    {
      "id": "matrix-profile/parent-float32-2048",
      "spec": {
        "side": "parent",
        "shape": [
          2048,
          2048,
          2048
        ],
        "dtype": "float32",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/device-night-20261026-v11/seeds/matmul_bias_activation.py",
        "source_sha256": "9d2cd2f13c6bd5c43cbbf7ce823e8175664948bd1752125b53982dffef834061"
      },
      "kernel": "mba_fp32_dual_v9_kernel_mix_aic",
      "duration_us": 334.426697,
      "algorithmic_flops": 17179869184,
      "paths": [
        {
          "component": "GM/L2",
          "path": "GM Read + Write",
          "official_bytes": 33562751.32290993,
          "official_flops": 68749108935.09671,
          "hardware_over_algorithmic": 4.001724820997142,
          "algorithmic_ai": 511.87308867235276,
          "algorithmic_tflops": 51.37110565069511,
          "reference_compute_tflops": 66.27840976514481,
          "reference_bandwidth_gbps": 1147.601937244727,
          "reference_bound_tflops": 66.27840976514481,
          "reference_achievement": 0.7750805402955018,
          "limiting_side": "compute",
          "reference_kind": "经验观察包络；不是已证明物理上界"
        },
        {
          "component": "GM/L2",
          "path": "L2 Read + Write",
          "official_bytes": 419790877.3283246,
          "official_flops": 68749108935.09671,
          "hardware_over_algorithmic": 4.001724820997142,
          "algorithmic_ai": 40.92482736484855,
          "algorithmic_tflops": 51.37110565069511,
          "reference_compute_tflops": 66.27840976514481,
          "reference_bandwidth_gbps": 4000.0,
          "reference_bound_tflops": 66.27840976514481,
          "reference_achievement": 0.7750805402955018,
          "limiting_side": "compute",
          "reference_kind": "官方L2带宽模型＋同精度经验有效计算参考"
        }
      ],
      "pipeline": {
        "aic_cube_ratio": {
          "mean": 0.119003703125,
          "minimum": 0.118997,
          "maximum": 0.119153,
          "count": 128
        },
        "aic_mte1_ratio": {
          "mean": 0.0549974609375,
          "minimum": 0.053718,
          "maximum": 0.056255,
          "count": 128
        },
        "aic_mte2_ratio": {
          "mean": 0.075251546875,
          "minimum": 0.043901,
          "maximum": 0.119066,
          "count": 128
        },
        "aic_scalar_ratio": {
          "mean": 0.0173559765625,
          "minimum": 0.014349,
          "maximum": 0.027623,
          "count": 128
        },
        "aic_scalar_mte1_stall_time(us)": {
          "mean": 36.8961268515625,
          "minimum": 35.381817,
          "maximum": 39.201214,
          "count": 128
        },
        "aiv_vec_ratio": {
          "mean": 0.0057141875000000005,
          "minimum": 0.005714,
          "maximum": 0.005747,
          "count": 256
        },
        "aiv_scalar_ratio": {
          "mean": 0.003990984375,
          "minimum": 0.000798,
          "maximum": 0.01366,
          "count": 256
        }
      },
      "issues": [
        "官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界"
      ]
    },
    {
      "id": "matrix-profile/pytorch-float16-1024",
      "spec": {
        "side": "pytorch",
        "shape": [
          1024,
          1024,
          1024
        ],
        "dtype": "float16",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/device-night-20261026-v11/seeds/matmul_bias_activation.py",
        "source_sha256": "9d2cd2f13c6bd5c43cbbf7ce823e8175664948bd1752125b53982dffef834061"
      },
      "kernel": "MatMulV2_ND_ND_FP16_FP16_false_false_all_98898",
      "duration_us": 16.380327,
      "algorithmic_flops": 2147483648,
      "paths": [
        {
          "component": "GM/L2",
          "path": "GM Read + Write",
          "official_bytes": 4194432.133944398,
          "official_flops": 2147483714.577439,
          "hardware_over_algorithmic": 1.0000000310025359,
          "algorithmic_ai": 511.98435912718656,
          "algorithmic_tflops": 131.10139058884477,
          "reference_compute_tflops": 248.59945314939048,
          "reference_bandwidth_gbps": 1147.601937244727,
          "reference_bound_tflops": 248.59945314939048,
          "reference_achievement": 0.5273599315202926,
          "limiting_side": "compute",
          "reference_kind": "经验观察包络；不是已证明物理上界"
        },
        {
          "component": "GM/L2",
          "path": "L2 Read + Write",
          "official_bytes": 25171968.029662285,
          "official_flops": 2147483714.577439,
          "hardware_over_algorithmic": 1.0000000310025359,
          "algorithmic_ai": 85.31250498449053,
          "algorithmic_tflops": 131.10139058884477,
          "reference_compute_tflops": 248.59945314939048,
          "reference_bandwidth_gbps": 4000.0,
          "reference_bound_tflops": 248.59945314939048,
          "reference_achievement": 0.5273599315202926,
          "limiting_side": "compute",
          "reference_kind": "官方L2带宽模型＋同精度经验有效计算参考"
        }
      ],
      "pipeline": {
        "aic_cube_ratio": {
          "mean": 0.674636375,
          "minimum": 0.652752,
          "maximum": 0.693797,
          "count": 16
        },
        "aic_mte1_ratio": {
          "mean": 0.5909654375,
          "minimum": 0.571418,
          "maximum": 0.613413,
          "count": 16
        },
        "aic_mte2_ratio": {
          "mean": 0.7808835625,
          "minimum": 0.732509,
          "maximum": 0.818776,
          "count": 16
        },
        "aic_scalar_ratio": {
          "mean": 0.408907,
          "minimum": 0.373826,
          "maximum": 0.448382,
          "count": 16
        },
        "aic_scalar_mte1_stall_time(us)": {
          "mean": 7.2560985,
          "minimum": 6.308485,
          "maximum": 8.14,
          "count": 16
        }
      },
      "issues": [
        "官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界"
      ]
    },
    {
      "id": "matrix-profile/pytorch-float16-2048",
      "spec": {
        "side": "pytorch",
        "shape": [
          2048,
          2048,
          2048
        ],
        "dtype": "float16",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/device-night-20261026-v11/seeds/matmul_bias_activation.py",
        "source_sha256": "9d2cd2f13c6bd5c43cbbf7ce823e8175664948bd1752125b53982dffef834061"
      },
      "kernel": "MatMulV3_ND_ND_ND_ND_FP16_FP16_FP16_FP16_high_performance_65536",
      "duration_us": 87.381744,
      "algorithmic_flops": 17179869184,
      "paths": [
        {
          "component": "GM/L2",
          "path": "GM Read + Write",
          "official_bytes": 16780032.0390402,
          "official_flops": 17179868739.970457,
          "hardware_over_algorithmic": 0.9999999741540789,
          "algorithmic_ai": 1023.8281514617817,
          "algorithmic_tflops": 196.60707600434253,
          "reference_compute_tflops": 248.59945314939048,
          "reference_bandwidth_gbps": 1147.601937244727,
          "reference_bound_tflops": 248.59945314939048,
          "reference_achievement": 0.7908588434673497,
          "limiting_side": "compute",
          "reference_kind": "经验观察包络；不是已证明物理上界"
        },
        {
          "component": "GM/L2",
          "path": "L2 Read + Write",
          "official_bytes": 201357311.48395005,
          "official_flops": 17179868739.970457,
          "hardware_over_algorithmic": 0.9999999741540789,
          "algorithmic_ai": 85.32031470518213,
          "algorithmic_tflops": 196.60707600434253,
          "reference_compute_tflops": 248.59945314939048,
          "reference_bandwidth_gbps": 4000.0,
          "reference_bound_tflops": 248.59945314939048,
          "reference_achievement": 0.7908588434673497,
          "limiting_side": "compute",
          "reference_kind": "官方L2带宽模型＋同精度经验有效计算参考"
        }
      ],
      "pipeline": {
        "aic_cube_ratio": {
          "mean": 0.82200085,
          "minimum": 0.805883,
          "maximum": 0.838992,
          "count": 20
        },
        "aic_mte1_ratio": {
          "mean": 0.69672765,
          "minimum": 0.678969,
          "maximum": 0.716059,
          "count": 20
        },
        "aic_mte2_ratio": {
          "mean": 0.7675335999999999,
          "minimum": 0.716245,
          "maximum": 0.819824,
          "count": 20
        },
        "aic_scalar_ratio": {
          "mean": 0.40420464999999994,
          "minimum": 0.39082,
          "maximum": 0.425325,
          "count": 20
        },
        "aic_scalar_mte1_stall_time(us)": {
          "mean": 10.623575650000001,
          "minimum": 8.783636,
          "maximum": 12.381818,
          "count": 20
        }
      },
      "issues": [
        "官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界"
      ]
    },
    {
      "id": "matrix-profile/pytorch-float32-1024",
      "spec": {
        "side": "pytorch",
        "shape": [
          1024,
          1024,
          1024
        ],
        "dtype": "float32",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/device-night-20261026-v11/seeds/matmul_bias_activation.py",
        "source_sha256": "9d2cd2f13c6bd5c43cbbf7ce823e8175664948bd1752125b53982dffef834061"
      },
      "kernel": "MatMulV2_ND_ND_FP32_FP32_false_false_all_66257",
      "duration_us": 39.840797,
      "algorithmic_flops": 2147483648,
      "paths": [
        {
          "component": "GM/L2",
          "path": "GM Read + Write",
          "official_bytes": 8388735.704127306,
          "official_flops": 9395240626.44014,
          "hardware_over_algorithmic": 4.374999844674086,
          "algorithmic_ai": 255.99610283864655,
          "algorithmic_tflops": 53.90162370496754,
          "reference_compute_tflops": 66.27840976514481,
          "reference_bandwidth_gbps": 1147.601937244727,
          "reference_bound_tflops": 66.27840976514481,
          "reference_achievement": 0.8132606665725087,
          "limiting_side": "compute",
          "reference_kind": "经验观察包络；不是已证明物理上界"
        },
        {
          "component": "GM/L2",
          "path": "L2 Read + Write",
          "official_bytes": 79044092.84365688,
          "official_flops": 9395240626.44014,
          "hardware_over_algorithmic": 4.374999844674086,
          "algorithmic_ai": 27.168173746361504,
          "algorithmic_tflops": 53.90162370496754,
          "reference_compute_tflops": 66.27840976514481,
          "reference_bandwidth_gbps": 4000.0,
          "reference_bound_tflops": 66.27840976514481,
          "reference_achievement": 0.8132606665725087,
          "limiting_side": "compute",
          "reference_kind": "官方L2带宽模型＋同精度经验有效计算参考"
        }
      ],
      "pipeline": {
        "aic_cube_ratio": {
          "mean": 0.9007604499999999,
          "minimum": 0.892266,
          "maximum": 0.91244,
          "count": 20
        },
        "aic_mte1_ratio": {
          "mean": 0.54034065,
          "minimum": 0.524747,
          "maximum": 0.552517,
          "count": 20
        },
        "aic_mte2_ratio": {
          "mean": 0.72326055,
          "minimum": 0.610034,
          "maximum": 0.778075,
          "count": 20
        },
        "aic_scalar_ratio": {
          "mean": 0.32087725,
          "minimum": 0.299998,
          "maximum": 0.345202,
          "count": 20
        },
        "aic_scalar_mte1_stall_time(us)": {
          "mean": 28.75190925,
          "minimum": 28.12303,
          "maximum": 29.511515,
          "count": 20
        }
      },
      "issues": [
        "官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界"
      ]
    },
    {
      "id": "matrix-profile/pytorch-float32-2048",
      "spec": {
        "side": "pytorch",
        "shape": [
          2048,
          2048,
          2048
        ],
        "dtype": "float32",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/device-night-20261026-v11/seeds/matmul_bias_activation.py",
        "source_sha256": "9d2cd2f13c6bd5c43cbbf7ce823e8175664948bd1752125b53982dffef834061"
      },
      "kernel": "MatMulV2_ND_ND_FP32_FP32_false_false_all_66256",
      "duration_us": 263.425262,
      "algorithmic_flops": 17179869184,
      "paths": [
        {
          "component": "GM/L2",
          "path": "GM Read + Write",
          "official_bytes": 33554556.80483186,
          "official_flops": 69793216009.07524,
          "hardware_over_algorithmic": 4.062499851516637,
          "algorithmic_ai": 511.9980956364799,
          "algorithmic_tflops": 65.21724246780859,
          "reference_compute_tflops": 66.27840976514481,
          "reference_bandwidth_gbps": 1147.601937244727,
          "reference_bound_tflops": 66.27840976514481,
          "reference_achievement": 0.9839892462553578,
          "limiting_side": "compute",
          "reference_kind": "经验观察包络；不是已证明物理上界"
        },
        {
          "component": "GM/L2",
          "path": "L2 Read + Write",
          "official_bytes": 440671741.4314816,
          "official_flops": 69793216009.07524,
          "hardware_over_algorithmic": 4.062499851516637,
          "algorithmic_ai": 38.98563844414615,
          "algorithmic_tflops": 65.21724246780859,
          "reference_compute_tflops": 66.27840976514481,
          "reference_bandwidth_gbps": 4000.0,
          "reference_bound_tflops": 66.27840976514481,
          "reference_achievement": 0.9839892462553578,
          "limiting_side": "compute",
          "reference_kind": "官方L2带宽模型＋同精度经验有效计算参考"
        }
      ],
      "pipeline": {
        "aic_cube_ratio": {
          "mean": 0.9847065,
          "minimum": 0.982748,
          "maximum": 0.986379,
          "count": 20
        },
        "aic_mte1_ratio": {
          "mean": 0.37772635,
          "minimum": 0.362679,
          "maximum": 0.390756,
          "count": 20
        },
        "aic_mte2_ratio": {
          "mean": 0.5583451,
          "minimum": 0.481741,
          "maximum": 0.606747,
          "count": 20
        },
        "aic_scalar_ratio": {
          "mean": 0.24493549999999997,
          "minimum": 0.23811,
          "maximum": 0.250694,
          "count": 20
        },
        "aic_scalar_mte1_stall_time(us)": {
          "mean": 196.86924275,
          "minimum": 195.595154,
          "maximum": 198.927277,
          "count": 20
        }
      },
      "issues": [
        "官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界"
      ]
    },
    {
      "id": "candidate-profile/candidate-float16-1024",
      "spec": {
        "side": "candidate",
        "shape": [
          1024,
          1024,
          1024
        ],
        "dtype": "float16",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/roofline-20261027/search-v2/single128x128-k256/candidate.py",
        "source_sha256": "a85e29936f435f78e026e2b84914c3dbee7cb3b42cf952dba011dee706133cbc"
      },
      "kernel": "hg_roof_single_tile_mix_aic",
      "duration_us": 43.200863,
      "algorithmic_flops": 2147483648,
      "paths": [
        {
          "component": "GM/L2",
          "path": "GM Read + Write",
          "official_bytes": 3146879.7839100785,
          "official_flops": 2154926000.658367,
          "hardware_over_algorithmic": 1.0034656155194934,
          "algorithmic_ai": 682.4168050460754,
          "algorithmic_tflops": 49.70927659477544,
          "reference_compute_tflops": 248.59945314939048,
          "reference_bandwidth_gbps": 1147.601937244727,
          "reference_bound_tflops": 248.59945314939048,
          "reference_achievement": 0.19995730467236272,
          "limiting_side": "compute",
          "reference_kind": "经验观察包络；不是已证明物理上界"
        },
        {
          "component": "GM/L2",
          "path": "L2 Read + Write",
          "official_bytes": 39927806.45347842,
          "official_flops": 2154926000.658367,
          "hardware_over_algorithmic": 1.0034656155194934,
          "algorithmic_ai": 53.78416293672743,
          "algorithmic_tflops": 49.70927659477544,
          "reference_compute_tflops": 248.59945314939048,
          "reference_bandwidth_gbps": 4000.0,
          "reference_bound_tflops": 215.1366517469097,
          "reference_achievement": 0.2310590789209374,
          "limiting_side": "memory",
          "reference_kind": "官方L2带宽模型＋同精度经验有效计算参考"
        }
      ],
      "pipeline": {
        "aic_cube_ratio": {
          "mean": 0.0601996875,
          "minimum": 0.059549,
          "maximum": 0.060803,
          "count": 64
        },
        "aic_mte1_ratio": {
          "mean": 0.059495109375,
          "minimum": 0.055307,
          "maximum": 0.065906,
          "count": 64
        },
        "aic_mte2_ratio": {
          "mean": 0.148705984375,
          "minimum": 0.072798,
          "maximum": 0.23408,
          "count": 64
        },
        "aic_scalar_ratio": {
          "mean": 0.032177609375,
          "minimum": 0.009035,
          "maximum": 0.102283,
          "count": 64
        },
        "aic_scalar_mte1_stall_time(us)": {
          "mean": 3.908787875,
          "minimum": 1.578788,
          "maximum": 7.106667,
          "count": 64
        },
        "aiv_vec_ratio": {
          "mean": 0.020803,
          "minimum": 0.020803,
          "maximum": 0.020803,
          "count": 128
        },
        "aiv_scalar_ratio": {
          "mean": 0.031937640625,
          "minimum": 0.005173,
          "maximum": 0.099183,
          "count": 128
        }
      },
      "issues": [
        "官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界"
      ]
    },
    {
      "id": "candidate-profile/candidate-float16-2048",
      "spec": {
        "side": "candidate",
        "shape": [
          2048,
          2048,
          2048
        ],
        "dtype": "float16",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/roofline-20261027/search-v2/single128x128-k256/candidate.py",
        "source_sha256": "a85e29936f435f78e026e2b84914c3dbee7cb3b42cf952dba011dee706133cbc"
      },
      "kernel": "hg_roof_single_tile_mix_aic",
      "duration_us": 174.243484,
      "algorithmic_flops": 17179869184,
      "paths": [
        {
          "component": "GM/L2",
          "path": "GM Read + Write",
          "official_bytes": 12585087.667042213,
          "official_flops": 17209638823.677,
          "hardware_over_algorithmic": 1.0017328210918348,
          "algorithmic_ai": 1365.0972991622925,
          "algorithmic_tflops": 98.59691042449542,
          "reference_compute_tflops": 248.59945314939048,
          "reference_bandwidth_gbps": 1147.601937244727,
          "reference_bound_tflops": 248.59945314939048,
          "reference_achievement": 0.3966095225690047,
          "limiting_side": "compute",
          "reference_kind": "经验观察包络；不是已证明物理上界"
        },
        {
          "component": "GM/L2",
          "path": "L2 Read + Write",
          "official_bytes": 293928956.0705092,
          "official_flops": 17209638823.677,
          "hardware_over_algorithmic": 1.0017328210918348,
          "algorithmic_ai": 58.44905317827484,
          "algorithmic_tflops": 98.59691042449542,
          "reference_compute_tflops": 248.59945314939048,
          "reference_bandwidth_gbps": 4000.0,
          "reference_bound_tflops": 233.79621271309935,
          "reference_achievement": 0.42172158941465665,
          "limiting_side": "memory",
          "reference_kind": "官方L2带宽模型＋同精度经验有效计算参考"
        }
      ],
      "pipeline": {
        "aic_cube_ratio": {
          "mean": 0.02976277734375,
          "minimum": 0.029572,
          "maximum": 0.030132,
          "count": 256
        },
        "aic_mte1_ratio": {
          "mean": 0.0317164921875,
          "minimum": 0.027948,
          "maximum": 0.033748,
          "count": 256
        },
        "aic_mte2_ratio": {
          "mean": 0.05037335546875,
          "minimum": 0.035455,
          "maximum": 0.103744,
          "count": 256
        },
        "aic_scalar_ratio": {
          "mean": 0.0066673359375,
          "minimum": 0.003581,
          "maximum": 0.025873,
          "count": 256
        },
        "aic_scalar_mte1_stall_time(us)": {
          "mean": 6.83680396484375,
          "minimum": 4.557576,
          "maximum": 13.795757,
          "count": 256
        },
        "aiv_vec_ratio": {
          "mean": 0.005141,
          "minimum": 0.005141,
          "maximum": 0.005141,
          "count": 512
        },
        "aiv_scalar_ratio": {
          "mean": 0.00435745703125,
          "minimum": 0.001202,
          "maximum": 0.022839,
          "count": 512
        }
      },
      "issues": [
        "官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界"
      ]
    },
    {
      "id": "candidate-profile/candidate-float32-1024",
      "spec": {
        "side": "candidate",
        "shape": [
          1024,
          1024,
          1024
        ],
        "dtype": "float32",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/roofline-20261027/search-v2/single128x128-k256/candidate.py",
        "source_sha256": "a85e29936f435f78e026e2b84914c3dbee7cb3b42cf952dba011dee706133cbc"
      },
      "kernel": "mba_fp32_dual_v9_kernel_mix_aic",
      "duration_us": 58.641174,
      "algorithmic_flops": 2147483648,
      "paths": [
        {
          "component": "GM/L2",
          "path": "GM Read + Write",
          "official_bytes": 8392832.565454496,
          "official_flops": 8597342290.233032,
          "hardware_over_algorithmic": 4.003449478295181,
          "algorithmic_ai": 255.87114138785486,
          "algorithmic_tflops": 36.62074787247609,
          "reference_compute_tflops": 66.27840976514481,
          "reference_bandwidth_gbps": 1147.601937244727,
          "reference_bound_tflops": 66.27840976514481,
          "reference_achievement": 0.5525290664371763,
          "limiting_side": "compute",
          "reference_kind": "经验观察包络；不是已证明物理上界"
        },
        {
          "component": "GM/L2",
          "path": "L2 Read + Write",
          "official_bytes": 54616063.648113206,
          "official_flops": 8597342290.233032,
          "hardware_over_algorithmic": 4.003449478295181,
          "algorithmic_ai": 39.319634271632246,
          "algorithmic_tflops": 36.62074787247609,
          "reference_compute_tflops": 66.27840976514481,
          "reference_bandwidth_gbps": 4000.0,
          "reference_bound_tflops": 66.27840976514481,
          "reference_achievement": 0.5525290664371763,
          "limiting_side": "compute",
          "reference_kind": "官方L2带宽模型＋同精度经验有效计算参考"
        }
      ],
      "pipeline": {
        "aic_cube_ratio": {
          "mean": 0.342853,
          "minimum": 0.342853,
          "maximum": 0.342853,
          "count": 32
        },
        "aic_mte1_ratio": {
          "mean": 0.1604531875,
          "minimum": 0.15488,
          "maximum": 0.165068,
          "count": 32
        },
        "aic_mte2_ratio": {
          "mean": 0.2739313125,
          "minimum": 0.152456,
          "maximum": 0.337315,
          "count": 32
        },
        "aic_scalar_ratio": {
          "mean": 0.0973700625,
          "minimum": 0.075371,
          "maximum": 0.153197,
          "count": 32
        },
        "aic_scalar_mte1_stall_time(us)": {
          "mean": 18.32874990625,
          "minimum": 16.026667,
          "maximum": 20.532122,
          "count": 32
        },
        "aiv_vec_ratio": {
          "mean": 0.0329079375,
          "minimum": 0.032905,
          "maximum": 0.033093,
          "count": 64
        },
        "aiv_scalar_ratio": {
          "mean": 0.033750234375,
          "minimum": 0.005026,
          "maximum": 0.07767,
          "count": 64
        }
      },
      "issues": [
        "官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界"
      ]
    },
    {
      "id": "candidate-profile/candidate-float32-2048",
      "spec": {
        "side": "candidate",
        "shape": [
          2048,
          2048,
          2048
        ],
        "dtype": "float32",
        "source": "/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/roofline-20261027/search-v2/single128x128-k256/candidate.py",
        "source_sha256": "a85e29936f435f78e026e2b84914c3dbee7cb3b42cf952dba011dee706133cbc"
      },
      "kernel": "mba_fp32_dual_v9_kernel_mix_aic",
      "duration_us": 335.806702,
      "algorithmic_flops": 17179869184,
      "paths": [
        {
          "component": "GM/L2",
          "path": "GM Read + Write",
          "official_bytes": 33562751.98012399,
          "official_flops": 68749110281.31772,
          "hardware_over_algorithmic": 4.001724899357518,
          "algorithmic_ai": 511.8730786490332,
          "algorithmic_tflops": 51.159994966389924,
          "reference_compute_tflops": 66.27840976514481,
          "reference_bandwidth_gbps": 1147.601937244727,
          "reference_bound_tflops": 66.27840976514481,
          "reference_achievement": 0.7718953298317438,
          "limiting_side": "compute",
          "reference_kind": "经验观察包络；不是已证明物理上界"
        },
        {
          "component": "GM/L2",
          "path": "L2 Read + Write",
          "official_bytes": 419790885.5485227,
          "official_flops": 68749110281.31772,
          "hardware_over_algorithmic": 4.001724899357518,
          "algorithmic_ai": 40.92482656347291,
          "algorithmic_tflops": 51.159994966389924,
          "reference_compute_tflops": 66.27840976514481,
          "reference_bandwidth_gbps": 4000.0,
          "reference_bound_tflops": 66.27840976514481,
          "reference_achievement": 0.7718953298317438,
          "limiting_side": "compute",
          "reference_kind": "官方L2带宽模型＋同精度经验有效计算参考"
        }
      ],
      "pipeline": {
        "aic_cube_ratio": {
          "mean": 0.118644375,
          "minimum": 0.118641,
          "maximum": 0.118719,
          "count": 128
        },
        "aic_mte1_ratio": {
          "mean": 0.054716109375,
          "minimum": 0.053379,
          "maximum": 0.056237,
          "count": 128
        },
        "aic_mte2_ratio": {
          "mean": 0.078866859375,
          "minimum": 0.043812,
          "maximum": 0.120469,
          "count": 128
        },
        "aic_scalar_ratio": {
          "mean": 0.0174111640625,
          "minimum": 0.014289,
          "maximum": 0.027698,
          "count": 128
        },
        "aic_scalar_mte1_stall_time(us)": {
          "mean": 37.023925171875,
          "minimum": 35.39212,
          "maximum": 39.237576,
          "count": 128
        },
        "aiv_vec_ratio": {
          "mean": 0.00569762109375,
          "minimum": 0.005697,
          "maximum": 0.00573,
          "count": 256
        },
        "aiv_scalar_ratio": {
          "mean": 0.00415379296875,
          "minimum": 0.000807,
          "maximum": 0.014086,
          "count": 256
        }
      },
      "issues": [
        "官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界"
      ]
    }
  ],
  "rules": [
    "PyTorch侧仅GEMM组件，候选为融合kernel；完整速度由KGS裁决。",
    "GM字节来自官方cache-miss/eviction模型，不能换成Memory.csv接口流量或逻辑字节。",
    "GM参考和矩阵点使用相同cache事件模型；逻辑copy吞吐单列。同精度经验线非物理峰值，官方0.8TB/s常量匹配未决。",
    "Vector内部操作不分配整个2MNK；FP32内部工作量与有效算法量分别报告。",
    "模型的limiting_side只表示参考线哪侧更低，不证明当前kernel计算受限。低达成率仍需IR/流水与受控实验区分原因。"
  ]
}


父版源码：
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generated by KernelGen: https://github.com/flagos-ai/KernelGen
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry

logger = logging.getLogger(__name__)


BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

# KLX quirk (2026-09-07): the xpu ttsdnnir pass chain rewrites any kernel
# whose symbol name starts with "matmul_bias_activation_kernel" into the
# sdnn hardware GEMM fusion, whose bf16 accumulation loses ~0.42 rms at
# K=4096 (exceeds the 1e-4*K tolerance; only bf16 is affected -- fp16/fp32
# stay within tolerance). Keep the fast sdnn path under the original name,
# and route large-K bf16 to the un-hijacked generic kernel. Remove this
# split once the vendor fixes sdnn accumulation precision.
SDNN_PRECISE_MIN_K = 4096


@libentry()
@triton.jit
def matmul_bias_activation_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    bias_ptrs = bias_ptr + offs_cn * stride_bias
    bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0)
    accumulator = accumulator + bias[None, :]

    # Apply ReLU activation
    # Workaround: synthesized zero tensor avoids a rank-0 constant src1
    # (NaN/-Inf edge differs from where-form; finite inputs identical)
    accumulator = tl.maximum(accumulator, accumulator * 0.0)

    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


@libentry()
@triton.jit
def fused_mba_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    bias_ptrs = bias_ptr + offs_cn * stride_bias
    bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0)
    accumulator = accumulator + bias[None, :]

    # Apply ReLU activation
    # Workaround: synthesized zero tensor avoids a rank-0 constant src1
    # (NaN/-Inf edge differs from where-form; finite inputs identical)
    accumulator = tl.maximum(accumulator, accumulator * 0.0)

    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_bias_activation(input, weight, bias):
    """
    Fused matmul + bias + ReLU activation.

    Args:
        input: Input tensor of shape (M, K)
        weight: Weight matrix of shape (K, N)
        bias: Bias vector of shape (N,) or (1, N)

    Returns:
        Output tensor of shape (M, N) with ReLU activation applied
    """
    assert input.shape[1] == weight.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (input.shape[0], weight.shape[1])
    ), "Incompatible input shape"
    M, K = input.shape
    _, N = weight.shape

    logger.debug("GEMS MATMUL_BIAS_ACTIVATION")
    if input.stride(0) > 1 and input.stride(1) > 1:
        input = input.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()
    if bias.dim() > 1:
        bias = bias.reshape(-1)
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # bf16 large-K: sdnn fusion accumulation exceeds tolerance, use the
    # generic (un-hijacked) kernel; everything else keeps the fast path.
    if input.dtype == torch.bfloat16 and K >= SDNN_PRECISE_MIN_K:
        kernel = fused_mba_kernel
    else:
        kernel = matmul_bias_activation_kernel

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(input.device):
        kernel[grid](
            input,
            weight,
            bias,
            out,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            weight.stride(0),
            weight.stride(1),
            bias.stride(0),
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )
    return out


run = matmul_bias_activation


_master_mba = matmul_bias_activation

@triton.jit
def mba_pipeline_kernel(
    A, B, Bias, C, M, N, K,
    SAM: tl.constexpr, SAK: tl.constexpr,
    SBK: tl.constexpr, SBN: tl.constexpr, SBIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ks = start * BK + rk
        ap = A + rm[:, None] * SAM + ks[None, :] * SAK
        bp = B + ks[:, None] * SBK + rn[None, :] * SBN
        if EVEN_M and EVEN_K:
            a = tl.load(ap)
        else:
            a = tl.load(ap, (rm[:, None] < M) & (ks[None, :] < K), 0)
        if EVEN_N and EVEN_K:
            b = tl.load(bp)
        else:
            b = tl.load(bp, (ks[:, None] < K) & (rn[None, :] < N), 0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
    bias = tl.load(Bias + rn * SBIAS, rn < N, 0)
    acc += bias[None, :]
    # 保留原版的 ReLU 表达式，兼容当前 Ascend 编译链。
    acc = tl.maximum(acc, acc * 0.0)
    tl.store(C + rm[:, None] * N + rn[None, :], acc,
             (rm[:, None] < M) & (rn[None, :] < N))


def _pipeline_run(input, weight, bias):
    assert input.ndim == 2 and weight.ndim == 2
    m, k = input.shape
    assert weight.shape[0] == k
    n = weight.shape[1]
    assert bias.numel() == n
    bias = bias.reshape(-1)
    out = torch.empty((m, n), device=input.device, dtype=input.dtype)
    tiny = m * n < 4096
    bm, bn, bk = (32, 32, 32) if tiny else (64 if m <= 512 else 128, 128, 128)
    # 真机分块诊断：低精度大矩阵增加 K 分块，减少循环和搬运同步。
    if input.dtype in (torch.float16, torch.bfloat16) and min(m, n, k) >= 1024:
        bk = 256
    with torch.npu.device(input.device):
        mba_pipeline_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            input, weight, bias, out, m, n, k,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1), bias.stride(0),
            bm, bn, bk, m % bm == 0, n % bn == 0, k % bk == 0,
            multibuffer=not tiny and k >= 2*bk, unit_flag=not tiny and k >= 2*bk, sync_solver=False,
        )
    return out





def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    # 非整齐形状使用原版，避免不规则边界的流水线问题。
    if m % 64 or n % 128 or k % 128:
        return _master_mba(input, weight, bias)
    return _pipeline_run(input, weight, bias)


matmul_bias_activation = run


@triton.jit
def mba_wide_pipeline_kernel(A, B, Bias, C, M: tl.constexpr, N: tl.constexpr,
                   K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    if GROUP == 0:
        pm, pn = pid // nn, pid % nn
    else:
        width = GROUP * nn
        first = (pid // width) * GROUP
        size = tl.minimum(nm - first, GROUP)
        pm = first + pid % size
        pn = (pid % width) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * K + rk[None, :]
    bp = B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        a = tl.load(ap, (rm[:, None] < M) & (rk[None, :] + start * BK < K), 0)
        b = tl.load(bp, (rk[:, None] + start * BK < K) & (rn[None, :] < N), 0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        ap += BK
        bp += BK * N
    acc += tl.load(Bias + rn, rn < N, 0)[None, :]
    acc = tl.maximum(acc, acc * 0.0)
    tl.store(C + rm[:, None] * N + rn[None, :], acc,
             (rm[:, None] < M) & (rn[None, :] < N))


_v4_mba = run
_WIDE_TILES = {('torch.bfloat16', 1024): (128, 128, 256, True, False, 0), ('torch.bfloat16', 2048): (128, 128, 256, True, False, 0), ('torch.bfloat16', 4096): (128, 128, 256, True, False, 0), ('torch.float16', 1024): (128, 128, 256, True, False, 0), ('torch.float16', 2048): (128, 128, 256, True, False, 0), ('torch.float16', 4096): (128, 128, 256, True, False, 0), ('torch.float32', 1024): (128, 128, 256, True, False, 0), ('torch.float32', 2048): (128, 128, 256, True, False, 0), ('torch.float32', 4096): (128, 128, 256, True, False, 0)}


def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    config = _WIDE_TILES.get((str(input.dtype), m))
    if (config is None or not (m == n == k) or not input.is_contiguous()
            or not weight.is_contiguous() or not bias.is_contiguous()):
        return _v4_mba(input, weight, bias)
    bm, bn, bk, multi, saving, group = config
    bias = bias.reshape(-1)
    out = torch.empty((m, n), dtype=input.dtype, device=input.device)
    with torch_device_fn.device(input.device):
        mba_wide_pipeline_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            input, weight, bias, out, m, n, k, bm, bn, bk, group,
            multibuffer=multi, enable_ubuf_saving=saving,
            unit_flag=True, sync_solver=False)
    return out


matmul_bias_activation = run


@triton.jit
def mba_selected_profiling_kgroup_v6_20261010_kernel(A, B, Bias, C, M: tl.constexpr, N: tl.constexpr,
                   K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    if GROUP == 0:
        pm, pn = pid // nn, pid % nn
    else:
        width = GROUP * nn
        first = (pid // width) * GROUP
        size = tl.minimum(nm - first, GROUP)
        pm = first + pid % size
        pn = (pid % width) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * K + rk[None, :]
    bp = B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        a = tl.load(ap, (rm[:, None] < M) & (rk[None, :] + start * BK < K), 0)
        b = tl.load(bp, (rk[:, None] + start * BK < K) & (rn[None, :] < N), 0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        ap += BK
        bp += BK * N
    acc += tl.load(Bias + rn, rn < N, 0)[None, :]
    acc = tl.maximum(acc, acc * 0.0)
    tl.store(C + rm[:, None] * N + rn[None, :], acc,
             (rm[:, None] < M) & (rn[None, :] < N))


_parent_profiling_kgroup_v6_20261010 = run
_TILES_profiling_kgroup_v6_20261010 = {('torch.bfloat16', 2048): (128, 128, 256, True, False, 4), ('torch.bfloat16', 4096): (128, 128, 256, True, False, 0), ('torch.float16', 2048): (128, 128, 256, True, False, 4), ('torch.float16', 4096): (128, 128, 256, True, False, 0), ('torch.float32', 2048): (128, 128, 256, True, False, 4), ('torch.float32', 4096): (128, 128, 256, True, False, 0)}


def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    config = _TILES_profiling_kgroup_v6_20261010.get((str(input.dtype), m))
    if (config is None or not (m == n == k) or not input.is_contiguous()
            or not weight.is_contiguous() or not bias.is_contiguous()):
        return _parent_profiling_kgroup_v6_20261010(input, weight, bias)
    bm, bn, bk, multi, saving, group = config
    bias = bias.reshape(-1)
    out = torch.empty((m, n), dtype=input.dtype, device=input.device)
    with torch_device_fn.device(input.device):
        mba_selected_profiling_kgroup_v6_20261010_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            input, weight, bias, out, m, n, k, bm, bn, bk, group,
            multibuffer=multi, enable_ubuf_saving=saving,
            unit_flag=True, sync_solver=False)
    return out


matmul_bias_activation = run


@triton.jit
def mba_dual_dispatch_v8_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Two N-adjacent tiles per program so the A panel is fetched once."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * (2 * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b0_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b1_ptrs = b0_ptrs + BLOCK_SIZE_N * stride_bn

    acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b0 = tl.load(b0_ptrs)
        b1 = tl.load(b1_ptrs)
        acc0 = tl.dot(a, b0, acc0, allow_tf32=False)
        acc1 = tl.dot(a, b1, acc1, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b0_ptrs += BLOCK_SIZE_K * stride_bk
        b1_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * (2 * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    base = c_ptr + stride_cm * offs_cm[:, None]
    bias0 = tl.load(bias_ptr + offs_cn * stride_bias)
    bias1 = tl.load(bias_ptr + (offs_cn + BLOCK_SIZE_N) * stride_bias)
    v0 = acc0 + bias0[None, :]
    v1 = acc1 + bias1[None, :]
    v0 = tl.maximum(v0, v0 * 0.0)
    v1 = tl.maximum(v1, v1 * 0.0)
    tl.store(base + stride_cn * offs_cn[None, :], v0.to(bias0.dtype))
    tl.store(base + stride_cn * (offs_cn + BLOCK_SIZE_N)[None, :], v1.to(bias1.dtype))

_v6_dispatch = run


def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    # 规则半精度矩阵一次复用 A 面板计算两个相邻 N tile。
    # 其余形状和 dtype 保留已完整通过的 v6。
    if not (input.dtype in (torch.float16, torch.bfloat16)
            and m % 128 == 0 and n % 256 == 0 and k % 256 == 0
            and input.is_contiguous() and weight.is_contiguous()
            and bias.is_contiguous() and bias.numel() == n):
        return _v6_dispatch(input, weight, bias)
    bias = bias.reshape(-1)
    out = torch.empty((m, n), device=input.device, dtype=input.dtype)
    with torch_device_fn.device(input.device):
        mba_dual_dispatch_v8_kernel[((m // 128) * (n // 256),)](
            input, weight, bias, out, m, n, k,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1), bias.stride(0),
            out.stride(0), out.stride(1), 128, 128, 256,
            multibuffer=True, unit_flag=True, sync_solver=False)
    return out


matmul_bias_activation = run


# ---------------------------------------------------------------------------
# v9: FP32 规则方阵的双 N tile 路径。
#
# 真机事实：float32::2（1024 方阵，69.5us vs 49.96us，0.718×）与
# float32::3（2048 方阵，358.4us vs 283.2us，0.790×）走的是 v8 之前的
# v6 单 N tile 路径（v8 只覆盖半精度），其真机流水为
# aic_mac 0.634 / aic_mte1 0.284 / aic_mte2 0.555 / aic_scalar 0.123，
# 即 MAC 活跃而标量与搬运仍有空档。
#
# 本路径只覆盖 fp32 且 m,n,k 均为 256 倍数（1024/2048/4096 方阵一定满足，
# 384 方阵一定不满足，走原回退），把 v8 “一次取 A 面板、两次消费”的结构
# 搬给 fp32：BM=128、BN=128、2 个 N tile、BK=128。
#
# 与失败的第 1 轮（v9 split 版）的关键区别：不再用 tl.reshape+tl.split
# 在 kernel 内部切 A，而是把 K 循环拆成两个 k 循环，每个带自己的、按
# BK=128 前进的 a_ptrs；两只 128×128 的 fp32 累加器按 N 方向相邻。
# 这样既不触发 tl.split 的编译期断言（该轮的真实失败点：
# `a_lo, a_hi = tl.split(tl.reshape(a, (BM, 2, BK)))` 处 AssertionError），
# 也避免 fp32 tile 因双 tile 翻倍导致的 UB 溢出。
#
# 半精度不受影响：本分支要求 fp32，半精度仍走上面的 v8 双 N tile 路径。
# ---------------------------------------------------------------------------


@triton.jit
def mba_fp32_dual_v9_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """fp32 规则方阵：每个 program 复用同一个 A 面板计算两个相邻 N tile。"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * (2 * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b0_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b1_ptrs = b0_ptrs + BLOCK_SIZE_N * stride_bn

    acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b0 = tl.load(b0_ptrs)
        b1 = tl.load(b1_ptrs)
        acc0 = tl.dot(a, b0, acc0, allow_tf32=False)
        acc1 = tl.dot(a, b1, acc1, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b0_ptrs += BLOCK_SIZE_K * stride_bk
        b1_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * (2 * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    base = c_ptr + stride_cm * offs_cm[:, None]
    bias0 = tl.load(bias_ptr + offs_cn * stride_bias)
    bias1 = tl.load(bias_ptr + (offs_cn + BLOCK_SIZE_N) * stride_bias)
    v0 = acc0 + bias0[None, :]
    v1 = acc1 + bias1[None, :]
    v0 = tl.maximum(v0, v0 * 0.0)
    v1 = tl.maximum(v1, v1 * 0.0)
    tl.store(base + stride_cn * offs_cn[None, :], v0.to(bias0.dtype))
    tl.store(base + stride_cn * (offs_cn + BLOCK_SIZE_N)[None, :], v1.to(bias1.dtype))


_parent_fp32_v9 = run


def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    # 只有 fp32 的规则方阵（m,n,k 均为 256 倍数）进入双 N tile 路径。
    # 半精度与一切不规则形状/步长继续走 v8 → v6 → master 的既有链路。
    if not (input.dtype == torch.float32
            and m % 256 == 0 and n % 256 == 0 and k % 256 == 0
            and input.is_contiguous() and weight.is_contiguous()
            and bias.is_contiguous() and bias.numel() == n):
        return _parent_fp32_v9(input, weight, bias)
    bias = bias.reshape(-1)
    out = torch.empty((m, n), device=input.device, dtype=input.dtype)
    with torch_device_fn.device(input.device):
        mba_fp32_dual_v9_kernel[((m // 128) * (n // 256),)](
            input, weight, bias, out, m, n, k,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1), bias.stride(0),
            out.stride(0), out.stride(1), 128, 128, 128,
            multibuffer=True, unit_flag=True, sync_solver=False)
    return out


matmul_bias_activation = run


_shared_edge_parent=run
def run(input,weight,bias):
    input=input.contiguous();weight=weight.contiguous();bias=bias.reshape(-1).contiguous()
    # 小K诊断域避开历史pipeline/双tile流水，使用既有通用数学内核。
    if input.shape[1] < 512:
        return _master_mba(input,weight,bias)
    return _shared_edge_parent(input,weight,bias)
matmul_bias_activation=run


通用候选空间与模板：
"""生成通用多输出分块；保持数学、整除门禁和未知布局父路径。"""
import ast

CONFIGS=[
 dict(id='dual-n-constexpr',bm=128,bn=128,bk=256,tm=1,tn=2,group=0,persistent=0,multi=True,stages=2,saving=False,interleave=False),
 dict(id='dual-m-reuse-b',bm=128,bn=128,bk=256,tm=2,tn=1,group=0,persistent=0,multi=True,stages=2,saving=False,interleave=False),
 dict(id='quad-64x128',bm=64,bn=128,bk=256,tm=2,tn=2,group=0,persistent=0,multi=True,stages=2,saving=False,interleave=False),
 dict(id='quad-128x64',bm=128,bn=64,bk=256,tm=2,tn=2,group=0,persistent=0,multi=True,stages=2,saving=False,interleave=False),
 dict(id='four-n-64',bm=64,bn=64,bk=256,tm=1,tn=4,group=0,persistent=0,multi=True,stages=2,saving=False,interleave=False),
 dict(id='dual-n-saving',bm=128,bn=128,bk=256,tm=1,tn=2,group=0,persistent=0,multi=True,stages=2,saving=True,interleave=False),
 dict(id='dual-n-interleave',bm=128,bn=128,bk=256,tm=1,tn=2,group=0,persistent=0,multi=True,stages=2,saving=False,interleave=True),
 dict(id='dual-n-group4',bm=128,bn=128,bk=256,tm=1,tn=2,group=4,persistent=0,multi=True,stages=2,saving=False,interleave=False),
 dict(id='dual-n-persistent20',bm=128,bn=128,bk=256,tm=1,tn=2,group=0,persistent=20,multi=True,stages=2,saving=False,interleave=False),
 dict(id='dual-n-stage1',bm=128,bn=128,bk=256,tm=1,tn=2,group=0,persistent=0,multi=True,stages=1,saving=False,interleave=False),
]

def validate(c):
    if set(c)!=set(CONFIGS[0]):raise ValueError('配置键不完整')
    if c['bm'] not in [64,128] or c['bn'] not in [64,128] or c['bk'] not in [128,256,512]:raise ValueError('分块越界')
    if c['tm'] not in [1,2] or c['tn'] not in [1,2,4] or c['tm']*c['tn']>4:raise ValueError('输出块数越界')
    if c['group'] not in [0,4] or c['persistent'] not in [0,20] or c['stages'] not in [1,2]:raise ValueError('流水配置越界')
    if any(type(c[k]) is not bool for k in ['multi','saving','interleave']):raise ValueError('布尔参数类型不符')

def generate(parent,c):
    validate(c);bm,bn,bk,tm,tn=[c[k] for k in ['bm','bn','bk','tm','tn']]
    body=['rk=tl.arange(0,BK)','nm=M//(BM*TM)','nn=N//(BN*TN)']
    if c['group']:
        body+=['width=4*nn','first=(tile//width)*4','group_size=tl.minimum(nm-first,4)',
               'pm=first+tile%group_size','pn=(tile%width)//group_size']
    else:body+=['pm=tile%nm','pn=tile//nm']
    for i in range(tm):body+=[f'm{i}=(pm*TM+{i})*BM+tl.arange(0,BM)',f'ap{i}=A+m{i}[:,None]*K+rk[None,:]']
    for j in range(tn):body+=[f'n{j}=(pn*TN+{j})*BN+tl.arange(0,BN)',f'bp{j}=B+rk[:,None]*N+n{j}[None,:]']
    for i in range(tm):
        for j in range(tn):body+=[f'c{i}{j}=tl.zeros((BM,BN),tl.float32)']
    loop=[]
    for i in range(tm):loop+=[f'a{i}=tl.load(ap{i})']
    if not c['interleave']:
        for j in range(tn):loop+=[f'b{j}=tl.load(bp{j})']
    for j in range(tn):
        if c['interleave']:loop+=[f'b{j}=tl.load(bp{j})']
        for i in range(tm):loop+=[f'c{i}{j}=tl.dot(a{i},b{j},c{i}{j},allow_tf32=False)']
    for i in range(tm):loop+=[f'ap{i}+=BK']
    for j in range(tn):loop+=[f'bp{j}+=BK*N']
    body+=['for kk in range(K//BK):']+['    '+x for x in loop]
    for j in range(tn):
        body+=[f'bias{j}=tl.load(Bias+n{j})']
        for i in range(tm):body+=[f'v{i}{j}=c{i}{j}+bias{j}[None,:]',f'v{i}{j}=tl.maximum(v{i}{j},v{i}{j}*0.0)',
            f'tl.store(C+m{i}[:,None]*N+n{j}[None,:],v{i}{j}.to(bias{j}.dtype))']
    definition='\n\n@triton.jit\ndef hg_mba_reuse_kernel(A,B,Bias,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,TM:tl.constexpr,TN:tl.constexpr):\n'
    if c['persistent']:
        definition+='    for tile in range(tl.program_id(0),(M//(BM*TM))*(N//(BN*TN)),tl.num_programs(0)):\n'
        definition+='\n'.join('        '+x for x in body)+'\n'
    else:definition+='    tile=tl.program_id(0)\n'+'\n'.join('    '+x for x in body)+'\n'
    wrapper=f'''
_hg_pipeline_parent=run
def run(input,weight,bias):
    m,k=input.shape;n=weight.shape[1]
    bm,bn,bk,tm,tn={bm},{bn},{bk},{tm},{tn}
    if not (input.dtype in (torch.float16,torch.bfloat16) and min(m,n,k)>=512
            and m%(bm*tm)==0 and n%(bn*tn)==0 and k%bk==0 and k>=2*bk
            and input.is_contiguous() and weight.is_contiguous() and bias.is_contiguous() and bias.numel()==n):
        return _hg_pipeline_parent(input,weight,bias)
    out=torch.empty((m,n),dtype=input.dtype,device=input.device)
    programs=(m//(bm*tm))*(n//(bn*tn))
    grid=min(programs,{c['persistent']}) if {c['persistent']} else programs
    with torch_device_fn.device(input.device):
        hg_mba_reuse_kernel[(grid,)](input,weight,bias.reshape(-1),out,m,n,k,bm,bn,bk,tm,tn,
            multibuffer={c['multi']},unit_flag=True,sync_solver=False,num_stages={c['stages']},enable_ubuf_saving={c['saving']})
    return out
matmul_bias_activation=run
'''
    text=parent+definition+wrapper;ast.parse(text);return text

def logical(c,m,n,k):
    bm,bn,tm,tn=[c[x] for x in ['bm','bn','tm','tn']]
    return dict(programs=(m//(bm*tm))*(n//(bn*tn)),
        logical_A_elements=m*k*(n//(bn*tn)),logical_B_elements=k*n*(m//(bm*tm)),
        accumulator_bytes=bm*bn*tm*tn*4,scope='源码逻辑请求与累加器下界；不是实际HBM／UB资源占用')


本轮实际已测结果（可能仍在筛选，未完成项目不推断收益）：
{
  "status": "running",
  "cases": [
    {
      "case": "dual-n-constexpr",
      "config": {
        "id": "dual-n-constexpr",
        "bm": 128,
        "bn": 128,
        "bk": 256,
        "tm": 1,
        "tn": 2,
        "group": 0,
        "persistent": 0,
        "multi": true,
        "stages": 2,
        "saving": false,
        "interleave": false
      },
      "status": "screened",
      "measurements": [
        {
          "size": 1024,
          "dtype": "float16",
          "times": {
            "parent": [
              0.028236666666666663,
              0.02808666666666667
            ],
            "candidate": [
              0.02723666666666667,
              0.02762666666666666
            ]
          },
          "speedup": 1.0266828957434453,
          "logical": {
            "programs": 32,
            "logical_A_elements": 4194304,
            "logical_B_elements": 8388608,
            "accumulator_bytes": 131072,
            "scope": "源码逻辑请求与累加器下界；不是实际HBM／UB资源占用"
          }
        }
      ],
      "source_sha256": "1dc27e33e785b2e4d502caf3e65a3972b380b24e4cf46270c602bfe4130ffaea",
      "screen_speedup": 1.0266828957434453
    },
    {
      "case": "dual-m-reuse-b",
      "config": {
        "id": "dual-m-reuse-b",
        "bm": 128,
        "bn": 128,
        "bk": 256,
        "tm": 2,
        "tn": 1,
        "group": 0,
        "persistent": 0,
        "multi": true,
        "stages": 2,
        "saving": false,
        "interleave": false
      },
      "status": "screened",
      "measurements": [
        {
          "size": 1024,
          "dtype": "float16",
          "times": {
            "parent": [
              0.028030000000000003,
              0.02802
            ],
            "candidate": [
              0.026606666666666664,
              0.026456666666666667
            ]
          },
          "speedup": 1.0562928504847133,
          "logical": {
            "programs": 32,
            "logical_A_elements": 8388608,
            "logical_B_elements": 4194304,
            "accumulator_bytes": 131072,
            "scope": "源码逻辑请求与累加器下界；不是实际HBM／UB资源占用"
          }
        }
      ],
      "source_sha256": "0c35cfbc23d9656cd47fb4168395f26928e6fbf0a3fe662a613a0636b877fb9c",
      "screen_speedup": 1.0562928504847133
    },
    {
      "case": "quad-64x128",
      "config": {
        "id": "quad-64x128",
        "bm": 64,
        "bn": 128,
        "bk": 256,
        "tm": 2,
        "tn": 2,
        "group": 0,
        "persistent": 0,
        "multi": true,
        "stages": 2,
        "saving": false,
        "interleave": false
      },
      "status": "running",
      "measurements": [],
      "source_sha256": "193ad660057ac8e44b25bba21b4eb0847a9334111bd58f37439b858bf69f8dbc"
    }
  ]
}


IR已核对：父版FP16存在两只128x128xf32累加器，A的128x256面板显式memref.copy一次，B两只256x128面板分别copy，随后两个linalg.matmul；文本次数不等于动态指令数。