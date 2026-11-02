# hardswish_out：严格Triton达标

`hardswish_out`计算分段线性激活，并将结果写入调用者提供的输出张量。本轮设备算法保留已验证的流式Triton核，新增C++宿主启动包装，减少每次调用中的Python元数据查询和参数对象组织。

|版本|相对CANN|相对PyTorch|正式评测|补测|实际后端|
|---|---:|---:|---|---|---|
|纯Python宿主＋流式Triton v1|0.776×|0.970×|两侧30/30|两侧42/42|已核验Triton|
|C++宿主＋同一Triton设备核 v4|0.862×|1.050×|两侧30/30|两侧42/42|已核验Triton|

直接比较两份已有正式测量中的候选耗时，整体几何平均提升为1.115×。小中输入通常从约15～17微秒降至13～14微秒，大输入约99～120微秒，变化较小。这与修改集中在固定宿主开销相符；未额外重测挑分。逐case数据及两次参考耗时变化见`proposals/hardswish_out/codex-vectorcall-out-v4/host-glue-effect.json`。

宿主桥每次重新读取当前输入、输出地址和stream，只缓存编译代码与元数据；缓存包含设备、输入／输出dtype及元素数。非连续、非对齐、空输入和计划缺失均回到原Triton入口。C++不执行数值计算、复制或CANN算子，不等于改为Ascend C。

本轮由Codex编写并逐函数审查宿主桥及Python入口，单CPU构建，保存源码／二进制／构建记录SHA。完整合同、布局与特殊值／新stream补测、运行时ATen检查、实际TTIR／npubin和设备trace均已核验。设备公式和流式核继承本批KG／CC／DeepSeek生成并验证的hardswish实现。

交付源码：`proposals/hardswish_out/codex-vectorcall-out-v4/main.py`；SHA256：`3d23f79ec3b0aeac20f7fa76c85bbfcb5efc3dc06946533ba046b384a258372c`。本项通过后严格Triton达标数为22/30；后续以动态交付清单为准。
