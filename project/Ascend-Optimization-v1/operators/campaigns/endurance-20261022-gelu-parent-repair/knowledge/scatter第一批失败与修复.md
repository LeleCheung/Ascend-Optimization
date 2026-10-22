# scatter_add第一批失败与第二批优先方向

第一批自包含候选修复了原版部分src/index长度的寻址错误，并尝试末轴连续快路径、小行稠密归约融合、减少启动参数。搜索最佳0.3471×，但独立语义17/18、性能输入数值7/9，不能交付。应优先复用下面的末轮源码，别从原版重复构建四轮：

/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/endurance-20261022/results/scatter_add/attempt-1/candidates/ab2327d7e898418d2b41966a67ec49a6d4152f359586a700cbec139a64b0242b/main.py

同目录semantics.json保存完整失败。全部第一批候选都涉及BF16重复索引累加误差：case2聚集索引时最大误差0.0546875，大BF16性能输入有少量0.03左右误差。请先以FP32暂存完成复制与原子累加，最后转回目标dtype，避免每次原子更新都在BF16舍入；一般stride和部分src长度路径也需相同精度保障。小行融合归约可直接在FP32求和，不必为它增加临时张量。该建议由Codex基于独立失败提出，仍需实测。

第一批实际metrics在research/scatter_add/search-round4-metrics-summary.json（原始附件在attempt-1/native-evidence归档）。大2048x2048的散射任务约19.5ms，MTE3活跃约97.5%，复制约20us；Vector很低。探针显示data-dependent atomic与无冲突地址写入都很慢，不能再仅靠BLOCK/num_warps枚举，也不能从MTE3高直接说HBM带宽饱和。第一批分析记录和debug-jobs包含对比依据。

在正确性先通过的前提下，可以继续研究分组/分段、亲和写入或本机已支持的更低层原子路径；不要调用PyTorch/FlagGems目标scatter_add，不升级FlagTree。若用TLE Raw须先核对当前版本能力及ABI。排序、分组和临时缓冲的总成本必须进入KGS完整调用评测，不能只展示局部kernel。

先通过check-endurance.py的18项及全部9项性能输入数值，每项两次和切stream；原合同和容差不改。真实有效源码即便仍低于0.8，也比有精度问题的高分版更适合作为交付和下轮起点。时间预算有限，第二批优先完成这一闭环。

此笔记在第一批worker结束后形成，只供第二批新overlay，实际贡献分别登记。
