# 真实证据与调用示例

| 示例 | 用途 | 是否需要设备 |
| --- | --- | --- |
| paired-expm1/ | 成对 CSV 聚合、源码身份核对与模型上下文 | 否 |
| matmul-instructions/ | 真实模拟器清单、热点筛选与源码区域聚合 | 否 |
| kgs-rad2deg/ | 严格 Triton 候选与 CANN 原生参考的完整 KGS 请求 | 是 |

离线样本来自冻结历史采集，保留原始文件与来源 SHA。expm1_ 是诊断案例，不表示该算子已达到 CANN 验收线。矩阵乘清单的源码身份由原采集请求核验后补入新封装，原指令记录保留在 original-listing.json。

KGS 示例包含同一已审查候选、36 项完整测试合同及参考库。设备评测前先确认环境、设备独占约定和服务状态；默认执行一次请求，不为挑选高分反复运行。

    hgkda kgs upload --server http://127.0.0.1:19661 --archive examples/kgs-rad2deg/operator.tar
    hgkda kgs evaluate --server http://127.0.0.1:19661 --request examples/kgs-rad2deg/request.json --review examples/kgs-rad2deg/review.json --run-directory /path/out/new-evaluation

候选的完整源码也保存在 benchmarks/operators/rad2deg_/candidate.py。服务启动和环境初始化属于可选后端部署，详见 docs/backends.md。
