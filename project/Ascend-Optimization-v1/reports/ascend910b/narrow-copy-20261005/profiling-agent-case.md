# narrow_copy 单 case profiling 诊断

- case：`benchmark/test_narrow_copy.py::test_narrow_copy_perf::core::float16::3`；设备：`Ascend910B4-1`
- 正式 KGS walltime：候选 101.10 us，PyTorch 22.06 us，加速比 0.218x。
- 真机 msprof：`_narrow_copy_flat_persistent_kernel` 设备时间 9.61 us；MTE2 0.691，MTE3 0.708，Vector 0.001，Scalar 0.129。
- AI Core simulator：指令 350 条，源码映射 90 条；候选源码行 [58, 68, 69, 71, 72]。

## 推断与下一步

候选是以数据搬运为主的拷贝 kernel，设备指令中的 MTE2/MTE3 值得核查；正式 walltime 与设备时间的差距提示应单独测量 host 启动、输出分配和同步。先为最差和最大 shape 分别采证，再只选择一个有证据的修改，并以完整 correctness 和 15 case walltime 复验。

Roofline：**unavailable**。缺少同一 kernel/range 的可信 FLOPs、实际搬运 Bytes 及 910B 实测 roof；纯拷贝不以计算 Roofline 判定上限。

## 证据边界

- Eval walltime 与 msprof/simulator 来自不同采集，不得相减解释为精确 host 时间
- instruction 采集可能包含输入生成 kernel；只使用映射到候选源码的行定位
- 单个 case 的证据不能推广为完整 15 case 的硬件结论

来源：`project\Ascend-Optimization-v1\reports\ascend910b\narrow-copy-20261005\eval-round-0001.json`、`project\Ascend-Optimization-v1\reports\ascend910b\narrow-copy-20261005\profiling-metrics.json`、`project\Ascend-Optimization-v1\reports\ascend910b\narrow-copy-20261005\profiling-instruction.json`；候选 SHA-256：`c31ed578ad12fb75957f44e0388210c9f3c4753c523de851e062f5afc7e4a99a`。
