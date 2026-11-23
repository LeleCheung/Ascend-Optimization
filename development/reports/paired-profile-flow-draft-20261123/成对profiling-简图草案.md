# 成对profiling配套草图

技术点1为成对profiling；技术点2为模拟器热点区域定位；技术点3为模拟器流水时序分析。两页均用正文26px、标题44px、可选脚本18px。

图上删除“同输入/同设备/同采集配置”和“瓶颈假设”方框。msprof op成对采集后，对比两组指标：数据搬运（总字节数/指令数/平均搬运粒度），流水活动（时间/活跃比例）。搬运量与粒度放进同一组，右侧Analyst Agent和Coder Agent下面单独连接淡蓝色编译验证小框，再进入迭代。

现有compare.py输出两侧原值、统计量、candidate_over_reference与derived_differences，还没有统一[0,1]归一化。图中差距标准化为方案，不修改现有研究后端或历史证据。

建议非负可比指标按d=abs(c-r)/max(c,r)编码差异大小，两侧均0时为0，缺失为未采得，明细保留raw_delta、candidate、reference、ratio、unit与aggregation，图中只写[0,1]。d=0.7不代表70%性能损失或瓶颈概率；对耗时/字节成本与流水活跃比例分别解释方向，不给一个万能好坏分数。每条具体指标分别编码，不将MTE时间、字节、比例混成单一0.7。

HTML提供真实GQA112MiB对64.296875MiB的差异编码，以及100对30得到0.7的示意。原始GQA数据来自development/reports/gqa-goal-20261121/gqa/iterations/attempt-006/paired-parent/comparison.json。归一化均为此草图新提出的计算，未声称历史Agent已使用。图中Analyst Agent与Coder Agent表示既有分析和实现职责，不额外声称独立多Agent编排。

等待/冲突从本页图中删除以突出主线，profiler仍可能提供对应汇总指标；模拟器增加的是具体时间位置、同步与重叠关系。采集入口collect_pair.py，比较compare.py，上下文与启发式假设context.py、diagnostics.py。Profiler比例形成排查方向，不能直接推出精确依赖；不同program数下的逐block均值保持各自范围。正式加速比使用独立真机评测。
