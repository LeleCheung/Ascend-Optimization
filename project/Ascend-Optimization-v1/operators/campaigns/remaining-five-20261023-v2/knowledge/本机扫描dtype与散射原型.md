# 本机扫描 dtype：源码证据与下一步验证

本批只读检查容器已安装 Triton：`tl.cumsum` 先把 BF16 升 FP32；FP16 默认不升级。`_pick_sum_dtype` 仅在显式 dtype 或小整数时选新 dtype。原始函数全文见同目录 installed-triton-scan-sort-source.json。

对 cumulative_trapezoid：上一轮 NPU 探针表明 FP16 增量直接 cumsum 与参考对齐，而先转 FP32 扫描不对齐。请优先在保持低精度 Add、乘法舍入后，对比 FP16 原 dtype 的 tl.cumsum 和显式 FP32。仍需独立检查结合顺序、跨块 carry 及后端 lowering；源码默认 dtype 不能直接证明设备累加机制或数值已正确。必须检查全部9项性能输入，不能改容差或调用框架 cumsum 作为候选核心。

对 scatter_add：同目录 scatter-sort-prototype.py 是 Codex 提出的尚未设备验证原型，scatter-sort-cpu.json 仅证明24组CPU数学模型在原容差内。候选使用本机 al.sort，排序 key 为 index*B+lane 的精确FP32编码，两 kernel 分隔排序/前缀与读取，避免直接依赖向量 gather 的旧设备问题。请先用正常 preflight/debug 验证编译、数值和性能，不把原型当作通过证据；如需要修改，保留原型原件与生成版本各自SHA。
