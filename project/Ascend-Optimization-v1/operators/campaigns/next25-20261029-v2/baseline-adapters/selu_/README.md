# selu_的KGS入口绑定

原冻结文件同时定义`selu_(*args, **kwargs)`和`run(A)`，两者不是同一函数。KGS要求同名公开符号与`run`绑定一致，因此在加载阶段报错，36项均未进入数值执行。

本目录将实现函数改名为`_fg_selu_impl`，相应调整`_original_selu`别名，保留`run(A)`；计算、分块、常量、布局处理和原有PyTorch回退保持原样。`baseline.py`按原源码AST做同一符号替换，再与新源码AST逐项比较，核验不仅依赖SHA。

重测已通过36/36，完整调用加速比0.135×。这是固定FlagGems原版的入口适配结果，属于原版列。

- 原源码SHA256：`49480fe1a22eeaa5a0dfcb0a3b5ff646875c5108dccb2857fca860b485bc9e8f`。
- 适配源码SHA256：`294d34200fbfe5652428146fe8c68b635adc3c3b4044dcd0e7f23f0c97210cdc`。
- 原失败：`results/selu_/baseline-fixed/FlagGems-1.result.json`。
- 重测：`results/selu_/baseline-entry-294d34200fbf/FlagGems-entry-1.result.json`。

旧文件和旧结果保留；新报告核验该入口适配后读取重测结果。
