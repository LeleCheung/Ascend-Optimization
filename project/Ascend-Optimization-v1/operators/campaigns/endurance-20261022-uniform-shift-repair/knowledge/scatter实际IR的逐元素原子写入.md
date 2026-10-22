# scatter_add：实际IR证实原子写入被逐元素展开

本批固定FlagGems父版、小型FP16代表输入的实际TTIR含tensor<256x...>的tt.atomic_rmw。后续ttadapter、bcmlir都出现带ExtractedLoadOrStore标注的scf.for，循环体把单个元素插入tensor<1xf16>，再用hivm.hir.store atomic=<add>写回。完整片段、原始IR路径和SHA见research/scatter_add/atomic-lowering-review.json及parent/ir/。

因此至少对这份编译实例，可以从源码数据依赖写地址落实到逐lane写回循环。结合首批大输入约19.5ms、MTE3高活跃以及无冲突atomic也慢的探针，第二批不应只搜索BLOCK/warps。流水活跃高不代表HBM达到峰值，减少原子冲突也未必消除这种地址降低成本。

优先保证BF16用FP32累加等语义。之后检查候选相同位置的实际IR是否仍有逐lane循环，再决定是否用按输出块归约、分组/排序或当前编译器已支持的低层数据搬运路径。每种替代均需完整计入预处理、暂存、转换与启动成本，不能调用PyTorch目标scatter_add。当前证据只证明父版代表实例，不宣称所有候选、dtype和输入均相同，也没有建立定量Roofline。

这是第一批结束后Codex的离线编译证据分析，提供给第二批新overlay，不改写第一批模型贡献。
