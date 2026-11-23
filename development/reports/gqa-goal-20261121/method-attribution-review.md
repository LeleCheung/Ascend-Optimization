# GQA方法来源复核与后续白盒实验

用户将“前期profiling、后期模拟器揭示具体问题并驱动多层修改与新增收益”列为最高优先级。主箭头表达具体改法的主要判断依据；实际工具读取、共同贡献、修改后重放分别保留。复核结果按源码SHA进入独立展示覆盖，原始决策、成绩与父链冻结。

| 主线节点 | 修改前记录 | 当前展示 |
| --- | --- | --- |
| 两项PV分解 | 源码／IR发现第三项的数值冗余；profiling与模拟器提供交接背景 | 保留共同参与与原历史入口 |
| KV64→128及因果修复 | 模拟器交接热点触发，profiling与IR支持；编译反馈修复因果UB | 橙色 |
| 编译期几何与掩码 | 宿主读取profiling；模型读取模拟器与TTIR，实际指令／掩码参与判断 | 橙色，共同贡献保留 |
| query行64→128 | profiling明确登记为trigger，模拟器登记为disambiguation，源码确认分块 | 蓝色*，事后主辅来源复核 |
| 单次FP16 PV、整块KV、KV复用、并行度与20worker均分 | 当前轮提供profiling与实际IR，宿主保留历史模拟器知识 | 蓝色；后重放单列 |

[复核数据与原件哈希](presentation-attribution-review.json) · [当前连续图](../gqa-goal-progress-20261121/index.html)

新实验032以029为父版本，修改前读取当前源绑定模拟器与实际TTIR／BCMLIR。Vector0的UB→OUT MOV_SRC_TO_DST_ALIGN为57.720001μs／16次，source_stack定位输出store；同处SET_FLAG时间与之重叠，单独解释。MTE3窗口productive39.39%，BCMLIR输出行stride8192，给出每行256B、行距16KiB的具体地址模式。成对profiling总字节量无法区分输出写与概率中间写，模拟器增加了具体写入位置的信息。

CC已实现仅PART_TILES输出物理BNSD布局，逻辑BSND保持；内核AST、20worker、KV复用和逐tile计算保持，地址映射双射审查通过，已进入原四项评测。返回张量逻辑非连续属性明确保存。此前023在019短程序上尝试同类布局的负结果继续保留，新调度下的假设由本轮设备结果与后重放判断。

新橙色主线节点采用完整原四项与修改后双分析验收后的真实收益。失败、持平、退步仍保存详情。031公共指数偏移原四项205.28415μs，目前后重放进行，数值组织的原合同限制与极端负值行下溢风险单列；它的具体改法由宿主profiling／源码假设提出，当前模拟器的实际读取保留为辅助参与。


2026-11-21用户下班前补充：首个成功节点137e585d…按连续两轮的profiling定位固定成本、测量反馈与源码去除冗余项这一联合路径改为蓝色*。round-02直接trigger仍明确为evaluation/source_ir，模拟器实际读取与background角色保留；这是展示层事后联合归因，未声称profiling独立主导。原复核文件另存presentation-attribution-review-before-first-node-20261121.json。KV分块等明确模拟器触发节点继续橙色。IR进入依据为具体源码表达／生成代码问题；当前未证明Triton整体性能上界，Triton搜索与已定位问题的IR实验继续。
