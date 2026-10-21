# 独立评测接口补测

<!-- 当前实测结果 -->
## 当前实测结果

- `baseline:silu_and_mul_masked`：passed，相对PyTorch 28.947×。
- 28.947×对应含expert循环和`.item()`同步的组合PyTorch参考；这是固定FlagGems成绩，不是本项目新增优化。

性能计时为完整调用walltime；来源、预算及逐case结果按原文件保留。下文是实验准备时的设计说明。

算法源码与原始失败均保留；ABI与用例身份见plan.json。
