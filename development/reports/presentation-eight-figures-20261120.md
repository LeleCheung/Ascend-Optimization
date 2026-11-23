# 四项算子的八张汇报图

IR阶段延续持续模拟器重放闭环：重放当前版本并定位热点与等待 → 选择Triton／IR修改层 → 编译修改版并再次重放 → 结合完整真机验收保留最佳。既有案例先完成真机评测，再采集修改后profiling及模拟器重放；执行顺序和窗口范围按原日志保留。详见[重放表述纠正](presentation-replay-correction-20261120.md)。

已将[组会网页](group-meeting-20261119/index.html)整理为两组八图：[四张全用例geomean进展](group-meeting-20261119/index.html#aggregate)和[四张固定shape机制曲线](group-meeting-20261119/index.html#fixed-shapes)。四项为BatchNorm、上采样、QKV和scatter_add。主线从原生FlagGems开始，沿已记录的进展连续连接完整合格最佳；下降、复测、失败、来源与源码继承保留在详情。

| 算子 | 当前全用例geomean / CANN | 当前固定shape / CANN | IR进展 |
| --- | ---: | ---: | --- |
| BatchNorm | 1.852× | 0.717× | 已验收 |
| 上采样 | 1.075× | 1.538× | 已验收 |
| QKV | 1.679× | 1.457× | 保留源码最佳，IR探索暂无新增收益 |
| scatter_add | 2.250× | 2.722× | 已验收 |

第一组直接使用26算子总览中对应四项的最新主图，同一份原始展示数据及逐case CANN参考。图标题显式标“全用例几何平均”，共同参考重算的点保留≈标记；同源码复测变化与代码收益分别说明。历史交付分数冻结保存，具体IR收益使用同合同原IR对比、二进制及ABBA验证。

第二组固定shape、dtype及case，标题显式标“固定shape”。每图下列出模拟器观察、成对profiling及其他诊断、模型具体改法、Triton或IR修改层、真机结果与证据范围。模拟器参与及Codex的机制分析按原记录区分；源码层和IR层的例子分别呈现。QKV完整IR负结果保留探索详情。

八图索引及来源SHA见[eight-figures.json](group-meeting-20261119/eight-figures.json)，机制说明数据见[case-evidence-notes.json](group-meeting-20261119/case-evidence-notes.json)。四个geomean SVG与26算子主图逐字节一致；26＋4图原生起点／连续／单调检查通过，八图截图及BatchNorm机制区检查通过，页面与表格横向溢出和链接检查通过，五项既有展示语义测试通过。

[持续重放修正版PPT](group-meeting-20261119/whitebox-simulator-ascend-20261120-replay.pptx)共22页，增加四页geomean曲线，保留四页固定shape曲线与机制材料。22页逐形状边界检查通过，桌面PowerPoint原生渲染尚未检查；原18页版与此前22页版按SHA保存在`group-meeting-20261119/history/`。本轮仅更新展示与讲稿，实验原件及成绩保持冻结。
