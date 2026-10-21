# 独立评测接口补测

<!-- 当前实测结果 -->
## 当前实测结果

- `baseline:scatter_add`：passed，相对PyTorch 0.258×。
- `baseline:upsample_bilinear2d`：failed，失败明细见results/。

性能计时为完整调用walltime；来源、预算及逐case结果按原文件保留。下文是实验准备时的设计说明。

算法源码与原始失败均保留；ABI与用例身份见plan.json。
