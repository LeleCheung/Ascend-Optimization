# 服务器集成验证工具

这里连接模型宿主、设备采集和KGS评测，不属于`hgkda`核心安装包。分析核心使用`python -m hgkda`；完整skill入口是[SKILL.md](../../skills/hgkda/SKILL.md)。

## 当前入口

从仓库根目录执行：

```bash
python -m tools.skill_validation --help
python -m tools.skill_validation prepare --help
python -m tools.skill_validation generate --config /path/to/run/config.json --invocation second
python -m tools.skill_validation context --config /path/to/run/config.json
python -m tools.skill_validation preflight --config /path/to/run/config.json
python -m tools.skill_validation evaluate --config /path/to/run/config.json
python -m tools.skill_validation pair --config /path/to/run/config.json --case log10_-timing-3
```

服务器操作需要Linux宿主、模型客户端及显式设备／容器配置；`--help`显示的动作不代表当前环境已具备相应后端。`prepare`只复制固定起点和测试合同，不改写候选；先检查回归计划及起点来源，再执行准备。

一次命令只执行一个动作。调用宿主仍负责选择取证、只读路径审查、将正式结果反馈给模型以及判断停止条件；此入口不等同于全自动外循环。候选修改由CC／DeepSeek完成，分析与调度工具不代写实现。

## 模块分工

| 部分 | 文件 | 用途 |
| --- | --- | --- |
| 准备与执行 | `prepare_operator.py`、`host_action.py`、`agent_runner.py`、`agent_tools.py` | 固定起点、单项调度、模型调用及有限工具桥 |
| 当前上下文 | `transfer_context.py` | 按配置绑定源码、合同、知识、IR、采集和正式反馈 |
| 成对采集 | `collect_pair.py`、`paired_driver.py`、`discover_kernel.py` | 发现实际kernel并按同输入采集候选和参考 |
| 编译与宿主诊断 | `runtime_api.py`、`compare_compiled.py`、`host_profile.py` | 实际接口、IR比较及合同驱动的函数栈采样 |
| 启动诊断 | `probe_launch.py`、`launch_probe.py` | 前者检查提交／同步及可选任务轨迹；后者检查冷／热缓存与调用钩子，二者不能互相替代 |
| 算子专项 | `tools/operators/selu_/`、`tools/operators/log10_/` | 专项检查按算子集中维护，统一动作仍由本入口调用 |
| 正式结果与证据 | `evaluate.py`、`audit_generation.py`、`finalize_transfer.py`、`export.py`、`verify.py` | 完整KGS、生成轨迹、方法冻结、导出和异机核验 |
| 历史兼容 | `legacy/recovery_context.py` | 仅支持早期SELU上下文／恢复合同；新实验使用`operator_transfer`配置 |

底层脚本可供调度器直接调用，但日常操作从统一入口选择动作。实验配置、模型设置、日志、缓存和新运行目录放在仓库外；结束后将冻结证据归档至`benchmarks/`。

## 旧入口恢复

2026-11-10移除`prepare_and_run.py`及`probe_host.py`：前者为早期SELU专用准备流程，包含旧的候选接口适配；后者固定FP16／1024输入，已由合同驱动的`host_profile.py`替代。旧上下文移入`legacy/`，专项检查最终集中到`tools/operators/<算子>/`。新工具保持现有测试合同及判定要求。

整理前完整版本是`08f6e11765ed3068c6234e9a10b7dfd1c0e2b917`。需要历史工具时，在独立目录检出该提交，或从对应实验的`method-snapshot/`读取原件，不把旧启动脚本恢复为日常入口。冻结证据仍使用当时文件路径及SHA，不随当前工具重命名。
