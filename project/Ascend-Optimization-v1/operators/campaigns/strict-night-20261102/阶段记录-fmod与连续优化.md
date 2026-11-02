# 严格Triton连续优化接续

本批已从冻结的16/30推进至21/30：原10项7项，新20项14项。新增为`hardswish`、`fix`、`hardshrink_backward`、`rad2deg_`、`fmod_tensor_`。性能验收是完整调用相对合格CANN参考超过0.8×，不要求每项已经超过1×。

## 新增fmod_tensor_

`fmod_tensor_`将两个张量逐元素相除后的截断余数写回第一个张量。本轮新增完整块流式Triton核，根据设备报告的Vector核数限制program数量，让每个program循环处理多个连续块；只有尾块带边界掩码。同时补齐编译缓存中的第二输入dtype，保留禁FMA、真实stride、特殊值和当前stream处理。

|相对CANN起点|当前相对CANN|当前相对PyTorch|正式评测|补测|运行时与实际后端|
|---:|---:|---:|---|---|---|
|0.7272×|0.8182×|0.7287×|两侧均30/30|两侧均57/57|19/19；TTIR、npubin和设备trace已核验|

源码及全部证据位于`proposals/fmod_tensor_/codex-streaming-fmod-v1/`，源码SHA256为`0f7bf1e0427e099f7820bc3bf89c7988a2f263d0d21842d9bd5e11cdc8ba3b43`。本提案由Codex直接实现并审查；KG／CC／DeepSeek的其他搜索结果保留各自归属。相对PyTorch仍不足0.8×，这里的达标使用用户固定的CANN参考口径。

## 当前接续任务

1. 完成`round_out`当前KG搜索，核验负零、最近偶数舍入、大幅值、decimals、stride及out语义。正式评测的高分在补测及实际后端核验前不计数。
2. 串行评测`log2_`的log10分块迁移与已审查宿主启动桥、`hardswish_out`的缓冲组织与out宿主桥，以及两个舍入算子的保符号快速公式。相同源码不重复计时挑分；已达标的待启动项保留现有成果。
3. 继续处理累计积分与求逆的数值问题，以及原生参考缺口；不改变冻结输入、容差或参考语义。

宿主桥只读取元数据、获取当前地址和stream、打包参数并调用Triton启动器。所有设备计算与复制仍为Triton。构建采用单CPU，源码／二进制／构建记录绑定SHA。CPU接口检查和公式检查仅用于在设备评测前发现错误。

服务器继续使用`hanle-strict-night-20261102`和项目物理卡7锁。恢复时先检查原会话、worker与请求终态，不重复启动；仅修改本项目的独立覆盖层与源码。

本文件为运行中阶段记录；最新达标数量及证据以本批`progress.json`、`delivery.json`与`严格Triton30项总表.md`为准。
