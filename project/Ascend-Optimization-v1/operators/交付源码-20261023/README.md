# 夜间最终交付源码

以下为正式评测所用源码的原字节副本，方便 Windows 直接打开。每份 SHA、正式结果和完整原始归档见[最终清单](../夜间连续优化交付-20261023.json)，方法与限制见[交付报告](../夜间连续优化交付-20261023.md)。

| 算子 | 源码 | 相对 PyTorch 完整调用加速比 |
|---|---|---:|
| `gelu_and_mul` | [main.py](gelu_and_mul/main.py) | 2.271× |
| `uniform` | [main.py](uniform/main.py) | 1.325× |
| `rad2deg_` | [main.py](rad2deg_/main.py) | 1.117× |
| `selu_` | [main.py](selu_/main.py) | 0.837× |
| `scatter_add` | [main.py](scatter_add/main.py) | 0.419×，未达标 |

这些是独立候选入口，不会自动替换安装的 FlagGems。运行依赖固定版本的容器环境、原批次合同及 KGS；不能直接拿本机 Python 执行代替完整评测。`cumulative_trapezoid` 未通过全部数值门禁，不提供合格交付源码。
