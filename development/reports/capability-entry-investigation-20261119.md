# 能力扩展入口核对

本记录为当前910B环境的只读接口调查，独立于三个单算子IR新增收益目标。原材料及示例源码保持原字节。接口调查完成后，再依据明确的表达缺口选择设备探针；目标设备代码继续由CC／DeepSeek编写。

## 当前安装入口

容器`tle_yy`、Triton3.5.1／FlagTree0.6.0、CANN9.0.0。已确认安装目录包含`triton/language/extra/cann/extension/`，导出`register_custom_op`、`custom`、`CORE`、`PIPE`、`MODE`及MLIR仿射映射接口；SDK包含bisheng及Ascend C头文件。

现有扩展包含`index_select_simd`、`gather_out_to_ub`、`scatter_ub_to_out`，builtin注册包含`__builtin_index_select`、`__builtin_gather_load`及`__builtin_scatter_store`。后续涉及普通gather／scatter的方案先检查这些现有入口。

## 双线性插值接口的观察

检查了CANN9.0.0的`BilinearInterpolation`公开接口与`dav_c220`实现。910B对应实现由Gatherb、Brcb、Mul、Add、Adds及Vector流水屏障组成；其中保留显式临时缓冲。该观察支持分析块收集、广播和同步组织，专用VBI单指令假设在当前910B头文件中未得到支持。`dav_m200`中的VBI调用属于不同架构的实现。

| 文件 | SHA256 |
| --- | --- |
| `tools/tikcpp/tikcfw/interface/kernel_operator_vec_bilinearinterpolation_intf.h` | `07401458f7baeb2bcbc08b513356124514f46ebcc2f5dcfa5b6238b5801d3a53` |
| `tools/tikcpp/tikcfw/impl/dav_c220/kernel_operator_vec_bilinearinterpolation_impl.h` | `8d81f9ed7dcbfdfaab08b3f9a997c3957124371a93eda046b87543b7a6bd67ce` |

下一项具体调查为UB块收集Gatherb及其同步组织：对照当前Triton／编译器对等价索引的实际lowering，再判断直接硬件表达是否缺失。当前先保存接口与架构区别，尚未生成能力扩展候选。

进一步核对当前安装源码：`gather_out_to_ub`的公开语义为GM到UB，`__builtin_gather_load`注册为SIMT；`index_select_simd`同为GM到UB，且要求索引维度位于末维之前。910B插值头文件中的Gatherb从UB块执行收集，上述接口各自的存储空间与索引范围分别记录。后续核对`tl.gather`及当前后端对UB收集的实际lowering，再决定custom op设备探针。当前尚未确认新增表达缺口。

## 本地材料

材料根目录为`E:/0_XSAI/00_KernelAgent/0_HGKDA_docs/BAAI-reports/05_work/custom_op/custom_op/`。已读取README、`npu_user_custom_op_ascendc.py`、`my_user_add_ascendc.cpp`与`triton-ascend-custom-op/SKILL.md`。原材料记录Triton3.6.0／CANN9.2.0的真机验证；其编译路径和版本与当前运行环境分别记录。

ABI关注项为输入在前、输出在后；UB memref描述符的地址、offset和stride；Vector操作与DMA之间的同步。具体操作的握手与缓冲需求根据当前编译产物验证。环境及示例验证结果按各自原件报告。
