# SELU 独立 skill 生成验证

从固定 FlagGems 实现开始，CC／DeepSeek 加载 HG-KDA skill、读取同输入 CANN profiling 与实际 IR、生成并修复严格 Triton 候选，Codex 完成调度、工具修复、源码审查及正式 KGS 提交。未提供历史最佳源码。

| 版本 | 相对 CANN | 正式合同 | 结论 |
| --- | --- | --- | --- |
| 历史交付源码经独立入口回归 | 1.039292× | 36/36 | 保留交付 |
| 本轮实际生成 | 0.210037× | 36/36；双方各42/42补测 | 性能未达0.8×，不替换交付 |

当前原30项合同仍达标26/30。本轮验证生成与评测衔接，不构成独立 skill 重现历史最佳性能的证据。额外偏移视图／周边存储检查由3/9修复至9/9；入口改名版设备路径完全相同，等价证据单列。

| 原件 | 内容 |
| --- | --- |
| [summary.json](summary.json) | 实验范围、结果、模型调用与代理返回的用量；费用字段不作为实际账单 |
| [workspace/starter-provenance.json](workspace/starter-provenance.json) | 固定起点、源码摘要与未提供历史最佳的记录 |
| [workspace/comparison.json](workspace/comparison.json)、[context.json](workspace/context.json) | 新采集的同输入对照及完整上下文 |
| [baseline-ir/](baseline-ir/) | 起点的实际编译产物与身份清单 |
| [attempts/](attempts/) | 四份设备预检源码、编译记录及检查结果 |
| [正式评测](formal-3cec1212c3b7/operators/selu_/evaluation/result.json) | 完整36项及逐case延迟 |
| [入口失败原件](formal-f3502fad5259/operators/selu_/evaluation/result.json) | 首份源码在KGS加载阶段发生公共符号冲突，未作为性能样本 |
| [source-review-entry-repaired.json](source-review-entry-repaired.json) | 最终源码逐路径审查 |
| [workspace/diagnosis.json](workspace/diagnosis.json) | 原始模型诊断，包含未证实机制，不能直接作为事实引用 |
| [workspace/final-feedback.md](workspace/final-feedback.md) | 正式结果反馈及对原诊断的修正 |
| [agent-calls/](agent-calls/)、[execution-snapshots/](execution-snapshots/) | 调用提示词、响应、用量、控制器快照和失败轨迹 |
| [windows-audit.json](windows-audit.json) | Windows离线原件核验，包含两份不同源码的正式请求 |

首次采集错误筛选 `Elu`，按真实日志改为 `Selu` 后只补采参考侧。模型宿主原有1%自动压缩设置造成循环，调用级设置修正为80%，全局设置未变。编译常量、视图偏移和KGS入口错误均保留失败原件；第四次设备预检仅因入口兼容修复显式增加，不进行同源码重复挑分。

在仓库根目录离线复核，使用已安装 HG-KDA 的 Python，输出换成新文件：

```bash
python tools/skill_validation/verify.py --directory benchmarks/skill-validation/20261109-selu --output /tmp/selu-skill-audit-new.json
```

Windows将输出路径换成仓库外的新文件。核验器在内存将外部冻结合同与检查器映射到 `benchmarks/regressions/20261109/fixed-25/`，不改原件、不执行设备任务。运行时依赖与镜像仍由开发环境配置；本目录保存证据，不包含venv或编译缓存。
