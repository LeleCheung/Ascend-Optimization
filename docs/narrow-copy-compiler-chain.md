# narrow_copy Ascend 编译链与调用开销报告

## 当前结论

`narrow_copy` 的主要问题不能概括成“Python 比 Triton 快”。Triton kernel 已经在 NPU 上执行；在小尺寸 case 中，设备 kernel 只有几微秒，而完整调用约百微秒，差距主要出现在输出分配、host launcher、运行时参数封装和同步路径。首次编译还需要约 `570 ms`，但编译后缓存命中仍然保留每次调用的运行时开销。

这说明要分开处理两个问题：

1. **编译期**：首次调用需要生成和编译 NPUBIN；它只影响冷启动或新 specialization。
2. **运行期**：即使编译缓存命中，每次仍要创建输出、选择 launcher、提交任务并同步；这是当前小 case 的主要限制。

## 910B 上实际使用的链路

当前容器版本为 FlagTree `0.6.0+ascend.gitc286cba6`、torch-npu `2.9.0.post2`、CANN `9.0.0`。安装包中的 Ascend backend 文件是：

`triton/backends/ascend/compiler.py`

其 `add_stages` 明确声明默认链路：

```text
Triton Python
  -> TTIR
  -> TTAdapter / Linalg IR
  ->（可选）MLIR bytecode，再恢复为 MLIR
  -> BiShengIR / CANN bishengir-compile
  -> NPUBIN
  -> CANN runtime launcher
```

后端还支持 `compile_mode="simt_only"`，理论上走 `TTIR -> NPUBIN`，绕过 TTAdapter/Linalg 阶段。它不是内联汇编，也不是公开的 Ascend 指令编程接口。

## 分层实测

来自 910B `tle_yy` 容器的同进程测量见 [分解结果](../project/Ascend-Optimization-v1/reports/ascend910b/narrow-copy-combined-20261006/decompose-narrow-copy-v5.json)：

| case | 首次调用 | warm 完整调用 wall | 仅 `new_empty` wall | PyTorch wall |
| --- | ---: | ---: | ---: | ---: |
| 64×64 float32 | 约 554.8 ms | 111.8 us | 40.0 us | 100.6 us |
| 1024×65536 float16 | 约 7.8 ms | 242.1 us | 39.3 us | 116.9 us |

这些 host probe 不是 KGS walltime，不能把两列直接相减当作精确设备时间；但它们足以证明：小 case 的分配和 launcher 路径占主要比例，大 case 还叠加了实际搬运时间。

同一批归档的独立 `msprof` 采集显示：小尺寸 float16 kernel 的设备平均时间约 `2.44 us`，大尺寸 float16 persistent kernel 约 `89.56 us`，后者 MTE2/MTE3 比例分别为 `0.916/0.7125`。这些采集与 host probe 不是同一次计时，不能相减；它们只用于确认 NPU kernel 本身的量级。编译缓存验证还记录了第一次 `compiled=1, hits=0`，随后五次保持 `compiled=1`、`hits=5`、`disabled=0`，见 [cache-verification](../project/Ascend-Optimization-v1/reports/ascend910b/narrow-copy-combined-20261006/cache-verification.json)。

编译缓存里的低层 runner 可以在拿到精确 specialization key、并传入候选自身的 `BLOCK/EVEN` constexpr 参数后调用；v5 的 `direct_launcher_kind` 为 `compiled_cache`，小 case launcher wall 约 `87.2 us`，大 case 约 `253.0 us`。早期少传 constexpr 时曾得到：

```text
TypeError: function takes exactly 15 arguments (13 given)
```

大 case 的参数数量也随 specialization 变化。这个对象依赖私有 `_COMPILED` 字典、地址对齐 key 和内部 packed metadata，不能当作稳定的 Python/C ABI，也不能安全地当作“内联汇编入口”。

## 低级入口验证

我用最小向量复制 kernel 实测了 `compile_mode="simt_only"`，结果失败。FlagTree 生成的命令向当前 CANN BiShengIR 编译器传入：

```text
--enable-triton-ir-compile
--pure-simt
--num-warps=32
--threads-per-warp=32
--shared-mem-dynamic-size=122880
```

当前 `bishengir-compile 1.1.0`（LLVM `19.1.7`，commit `428ab8fdab46`）全部报告 unknown argument；它只提示了不同的参数名，例如 `--enable-triton-kernel-compile`。原始失败 JSON 见 [compile-mode 探针](../project/Ascend-Optimization-v1/reports/ascend910b/narrow-copy-combined-20261006/probe-ascend-compile-mode.json)。

实际 cache 中确实保存了一个 flat kernel specialization 的 `.source`、`.ttir`、`.ttadapter`、`.mlirbc`、`.bcmlir`、`.npubin` 和 metadata；样本及 SHA manifest 见 [narrow-copy-ir](../project/Ascend-Optimization-v1/reports/ascend910b/narrow-copy-combined-20261006/narrow-copy-ir/)。这些文件证明中间产物存在，但不等于它们构成稳定的外部编程接口；当前报告没有修改或替换这些内部产物。

因此当前结论是：

- **LLVM/Ascend IR 并非完全不存在**：后端明确加载 Ascend dialect，并能缓存 TTIR、TTAdapter、MLIR bytecode 和 NPUBIN 中间产物。
- **直接注入内部 IR 目前不具备可复现接口**：需要匹配 FlagTree、BiShengIR、CANN 三者的版本和内部 schema，仓库没有稳定的公开契约。
- **CUDA 式 inline asm 不可直接迁移**：现有 Triton Ascend DSL 没有可验证的 inline assembly 入口；底层指令应通过 Ascend C/TIK/CANN Custom Operator 等独立工具链实现。
- **当前最现实的优化方向不是手写 IR**：先减少输出分配/调度次数，或把多个小 copy 融合成一个调用；这直接针对已测出的运行期瓶颈。

## 下一项实验

在不改变算子语义的前提下，增加一个“预分配 output、复用已解析 plan、只提交一次 kernel”的内部 benchmark，并与现有 `run` 同卡交替测量。成功标准是：

1. correctness 全通过；
2. 小 case 的 warm walltime 明显低于约 `116 us`；
3. 结果能解释 KGS 与 host probe 的差异；
4. 不把内部 compiled runner ABI 暴露为项目正式接口。

如果该实验仍不能接近 PyTorch，才有充分证据把问题上移为 Ascend Triton/CANN runtime 路径差异，再评估 Ascend C/TIK custom operator，而不是直接修改内部 IR。
