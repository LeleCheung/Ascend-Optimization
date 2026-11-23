# 昇腾规格补充来源复核

2026-11-22。核对用户提供附件后半段新增的华为逐项说明，并直接读取其中四个官方页面。在现有910B容器的CANN 9.0.0 SDK中，另以CPU程序查询显式指定的Ascend910B4-1平台配置。比较表Excel和PDF保持原文件。

## 结论

现有分享尚未补齐Ascend910B4-1全部硬件规格。七项硬件信息继续建议为：L1/SPM、core数量标√；寄存器、L2、DRAM、矩阵算力、向量算力标部分。依据采用具体型号匹配及该行子项覆盖情况。

| 指标 | 补充材料提供的依据 | 910B4-1判定依据 |
| --- | --- | --- |
| 寄存器数量/大小 | SIMD寄存器文档给出256 B、数量上限32，以及32 B掩码寄存器、上限8 | 文档产品范围为950PR/950DT；910B4-1完整寄存器规格仍缺项，部分。 |
| L1/SPM容量/带宽/访问粒度 | GetCoreMemSize；已保存的910B SDK容量、UB block及官方L1通路模型 | 保持√，L1 Buffer与UB分别记录。 |
| L2容量/带宽/cache line size | GetCoreMemBw提供查询接口；架构说明列128/256/512 B随硬件变化 | 910B4-1配置容量168 MiB；API带宽参数110 B/cycle；cache line的型号映射仍未补齐，部分。 |
| DRAM容量/带宽 | 分享中的204.8 GB/s、24 GB属于Atlas 300I Pro；GetCoreMemBw另支持HBM | 已取得910B4-1 SDK带宽参数32 B/cycle；整卡HBM峰值及其与当前板卡的对应仍待核实，部分。 |
| 4/8/16/32 bit矩阵算力 | 300I Pro公开INT8 140 TOPS、FP16 70 TFLOPS；分享明确完整分表尚缺 | 910B4-1完整精度峰值表仍未补齐，部分。 |
| 4/8/16/32 bit向量算力 | 分享明确同一产品完整精度峰值表尚缺 | 910B4-1仍缺匹配的完整峰值表，部分。 |
| core数量 | PlatformAscendC查询；既有运行时及SDK记录 | 20 Cube、40 Vector，保持√。 |

## 本轮SDK接口查询

使用`PlatformAscendCManager::GetInstance("Ascend910B4-1")`明确选择SoC配置，调用`GetCoreMemBw`、`GetCoreMemSize`。CPU编译和程序执行成功，未运行NPU kernel。

| 类型 | GetCoreMemBw | GetCoreMemSize |
| --- | --- | --- |
| L2 | 110 B/cycle | 176160768 B，即168 MiB |
| HBM | 32 B/cycle | 34359738368 B，即32 GiB |

单位B/cycle来自官方API说明。这些值属于所选SDK平台配置。HBM容量与既有板卡查询65536 MB存在差异；整卡物理峰值的解释仍需匹配板卡与通路范围。保留既有官方Roofline默认/SK/SS与CJ分支值，分别注明模型来源。当前已取得带宽查询接口和参数，后续说明采用“整卡峰值及适用范围尚未核实”。

首次CPU链接缺少`libunified_dlog`，补充链接该库后编译及执行成功。原始结果分别保存在`sdk-bandwidth-api-result.json`、`sdk-bandwidth-api-result-linked.json`；SDK头文件与配置见`sdk-platform-query.json`。

## 本轮直接读取的官方来源

- [寄存器类型定义](https://asc.gitcode.com/api/SIMD-API/c_api/defs/type/data_type_definition.html)：适用于950PR/950DT系列的SIMD寄存器接口与规格；本地`RegisterTypes.json`。
- [GetCoreMemBw](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1035.html)：支持L2、HBM类型，输出单位byte/cycle；本地`GetCoreMemBw.json`。
- [Basic Architecture](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)：cache line为128/256/512 B，取值随硬件规格变化；本地`BasicArchitecture.json`。
- [Atlas 300I Pro技术规格](https://e.huawei.com/cn/products/computing/ascend/atlas-300-ai)：LPDDR4X 24 GB、204.8 GB/s、INT8 140 TOPS、FP16 70 TFLOPS；本地`Atlas300.json`。

既有910B4-1板卡、运行时、有效profiling字段及模拟器依据继续使用`../machine-indicator-audit-20261121/`、`../ascend-statistics-20261120/`和`../ascend-spec-public-20261121/`。
