# 算子专项工具

通用分析与评测控制器位于`src/hgkda/`、`tools/skill_validation/`和`tools/regression/`。本目录按算子集中维护专项正确性检查与CANN参考适配，不创建每个算子独立的优化控制器。

| 算子目录 | 工具与用途 |
| --- | --- |
| `selu_/` | 偏移视图、当前stream、指针与尾块检查 |
| `log10_/` | 极小值、特殊值、数学边界及尾块检查 |
| `negative_/` | 核验已有宿主缓存候选的实际Triton路径及动态输入；不生成候选 |
| `uniform/` | 构建和核验原生Uniform＋Clamp同语义参考 |
| `expm1_/` | 探测官方参考的精度模式与小值数学资格 |

SELU／log10补测仍可通过`python -m tools.skill_validation check-selu-offset`等统一动作调用。专项工具不会替代正式KGS合同。需要设备的命令仅在匹配的Linux／CANN环境及指定项目设备上执行。

直接拼接或替换候选的旧构建脚本已移至[历史工具](../archive/README.md)，后续候选及其启动封装均由CC／DeepSeek修改。
