# narrow_copy 最终归因与方案决策

## 结论

当前不能把问题简单说成“Python 比 Triton 快”，也不能笼统归因成“Ascend Triton 编译器不行”。已经把冷编译、编译缓存、输出分配、launcher、同步和 NPU kernel 分开测量后，结论是：

- 首次编译是冷启动成本：小 case 约 `555 ms`，大 case 约 `7.8 ms`；编译缓存命中后不会每次重复支付。
- 小 case 的 NPU kernel 约 `2.44 us`，而完整 warm 调用约 `112 us`。主要差距在 output 分配、host 参数封装、launcher 和同步路径。
- 大 case 预分配后 `run_into` 约 `257.5 us`，PyTorch native 约 `162.5 us`；此时除了 host/runtime 差距，还包含 Triton 生成 kernel 的搬运路径差异。
- 预分配、plan 复用和 compiled specialization 不是无效方向：同一进程的实验性 `run_into` 把小 case 从 `116.8 us` 降到 `104.7 us`，大 case 从 `291.6 us` 降到 `257.5 us`。但它仍不是 FlagGems Definition 的公开 ABI，不能直接作为最终算子接口。

## 完整验证

加入实验性 `run_into` 的候选仍通过 KGS `33/33`（18 correctness、15 timing），几何平均 `0.5128×`；独立边界用例 `27/27` 通过。该接口只用于测量运行时因素，没有改变正式 `run(inp, dim, start, length)` ABI，也没有修改共享 FlagGems 或 KGS `19650`。

## 编译链定位

当前 910B 容器版本为 FlagTree `0.6.0+ascend.gitc286cba6`、torch-npu `2.9.0.post2`、CANN `9.0.0`。实际链路是：

```text
Triton Python → TTIR → TTAdapter/Linalg IR → MLIR bytecode（可选）
→ BiShengIR/CANN → NPUBIN → CANN runtime launcher
```

cache 中能看到 `.ttir`、`.ttadapter`、`.mlirbc`、`.bcmlir` 和 `.npubin`，说明中间产物确实生成。但它们由当前 FlagTree/CANN 版本内部 schema 约束，不能当作稳定外部接口。

`compile_mode="simt_only"` 的最小 kernel 实验失败，原因不是 kernel 语义，而是 FlagTree 传给当前 `bishengir-compile 1.1.0` 的参数不兼容：`--enable-triton-ir-compile`、`--pure-simt`、`--num-warps=32` 等被报告为 unknown argument。这个版本组合下，直接切换旁路不可复现。

## 低级路径决策

| 路径 | 当前判断 | 原因 |
| --- | --- | --- |
| 直接改 TTIR/内部 Ascend IR | 暂不采用 | 产物存在，但 schema、版本和 metadata 未形成稳定外部契约；容易得到不可复现的二进制 |
| Triton inline assembly | 不可用 | 当前 Ascend Triton DSL 中没有经过验证的公开入口 |
| 复用内部 compiled runner | 仅作实验 | 可用私有 `_COMPILED` key 调到，但依赖地址对齐和内部 packed ABI，不适合提交给导师作为产品方案 |
| Ascend C/TIK/CANN Custom Operator | 可行性最高 | 是公开/半公开的低级开发方向，但需要单独构建 operator、注册 tiling/metadata，并承担新的调用封装成本 |
| CANN `aclrtMemcpyAsync` D2D | 已验证可调用 | 910B 上最小 C++ smoke test 返回成功；但它只证明 runtime memcpy API 可用，尚未证明能嵌入 FlagGems Definition 或胜过 PyTorch |

## 方案决策

保留当前 Triton 候选作为可复现基线，同时保留实验性 `run_into` 作为运行时归因工具，不把它伪装成正式 Definition 优化。对于当前 workload，最现实的后续实验是写一个独立 C++/CANN D2D memcpy benchmark，与预分配 Triton 和 PyTorch 在同一输入、同一 stream、同一同步口径下比较；成功标准是完整 correctness 不变且大 case walltime 接近或低于 `162.5 us`。只有这个实验成功，才值得把 CANN native/custom op 接入算子 ABI。

本案例的可交差结论是：**Ascend Triton 编译链不是唯一核心限制。冷编译只影响首次调用；warm 场景的主要问题是 Triton 算子接口带来的 output/launcher/runtime 成本，小 case 尤其明显，大 case 还叠加了 kernel 搬运效率差距。直接写 IR 或 inline asm 目前没有稳定、可复现的入口；CANN native/custom operator 是下一条合理路线。**
