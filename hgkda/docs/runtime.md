# 公共运行接口

安装本目录Python包后，分析入口是python -m hgkda，模型与设备集成入口是python -m hgkda.runtime。后者随包安装；不读取开发仓库中的工具。

## 准备

环境JSON由宿主提供container、device、device_python、sdk、lock、kgs_server。容器需能访问交付目录、运行目录以及HG-KDA设备驱动文件；设备解释器需安装匹配的Triton、torch_npu和CANN环境。KGS是外部独立服务，公共入口不自动启动或重启它。

    python -m hgkda.runtime prepare --operator-directory /path/operators/log10_ --package /path/hgkda --environment /path/environment.json --run-directory /path/runs/new-run
    python -m hgkda.runtime context --config /path/runs/new-run/config.json
    python -m hgkda.runtime preflight --config /path/runs/new-run/config.json
    python -m hgkda.runtime evaluate --config /path/runs/new-run/config.json

prepare以交付清单中已选用源码为起点，原字节复制并声明prior_best_source_provided=true，不把该流程当作从上游从零生成。原正式请求与计时设置保留；仅为交付中已重定位的参考重新打包合同并记录新哈希。

## 采集与模型

    python -m hgkda.runtime pair --config /path/runs/new-run/config.json --case log10_-timing-3
    python -m hgkda.runtime simulate --help
    python -m hgkda.runtime generate --config /path/runs/new-run/config.json --invocation iteration-1

生成适配器当前支持本机Claude Code命令，由run目录中的model-settings.json选择模型和凭据；该文件由宿主另行提供，不随发布或实验归档。生成与设备操作各执行一个明确动作，宿主负责只读源码审查、多轮编排及停止判断。不要将这些单项入口表述为完整无人值守调度器。

默认`optimization_policy=performance_first`：不设总搜索预算，达到验收阈值不自动结束优化。`generate`默认`--max-turns 0 --generation-timeout 0`，分别表示不向宿主传入工具轮数上限、不设置整次生成截止时间。模型返回后由宿主继续评测、取证和下一轮；不使用固定失败次数终止。仅任务明确要求有限预算时传正数，或明确要求达标即止时配置`optimization_policy=target_only`。

这些设置不移除模型上下文／单次输出容量，也不取消设备单项操作的防挂死超时、源码审查与设备锁。前者需要分段续接，后者用于记录异常和恢复执行，均不是总优化预算。冻结实验的历史预算和计时合同保持原样。

候选和原生参考的同一次API调用均可以包含多个设备任务。成对采集分别保存这些实际任务及各自CSV，比较器保留逐kernel分布并按调用次数汇总计数；二进制符号以本次采集记录为依据。多kernel侧的单一Task Duration保留未知，完整调用时间由正式评测另行取得。带时间戳后缀的CSV按逻辑表名识别，同时保留原文件SHA。

多kernel候选的模拟器采集通过`python -m hgkda.runtime.collect_simulator --kernel-name <已发现符号>`显式选定目标，其他参数沿用单kernel采集接口。选择须来自当前源码和用例的发现列表；调用方记录选取理由。各kernel独立采集与分析，整体收益仍通过完整算子评测确认。

模拟器trace单位通过同PC的Vector／搬运等非同步指令与CSV交叉核验。Scalar的CSV／trace量化口径可能不同，单列对照而不强行拟合；时间轴仍保留工具实际报告区间。同步等待和单核空闲的解释边界见skill接口。

调用前保存源码审查到run/source-reviews/<SHA>.json。生成模型修改源码后，旧审查不再适用。完整性能反馈进入下一轮上下文，未完成正式评测不能以预检代替验收。配置允许device_pythonpath显式提供设备侧外部依赖，不默认使用内部FlagGems快照。

## 分析与复现

新实验自动保存trace事件和版本／结果引用；具体修改前用`python -m hgkda.runtime decision`登记显式依据，最后用`python -m hgkda.runtime trace`导出连续事件JSONL。文件结构、参数及“作者声明不等于因果证明”的边界见[连续trace](../skills/hgkda/references/trace.md)。原始模型工具日志继续保留。

比较、热点、区域和上下文命令见[Skill接口](../skills/hgkda/references/interface.md)。公开包内知识是完整安装数据；记录中历史来源链接是追溯信息，不在运行时加载开发档案。算子补测通过operators/<name>/checks与references执行。

源码和包的单元测试属于development/tests/hgkda，外部skill不会调用它们。用户无需提供development目录。

## FlagGems 原生验收

    python -m hgkda.runtime upstream-test --config /path/upstream-test.json

这是独立于KGS的可选交付步骤。配置指向外部FlagGems Git工作区、固定commit、接入文件及其SHA、设备解释器和新输出目录；公共包不导入内部development快照。接入后的源码由模型生成并单独审核。测试器确认实际导入的目标实现，调用上游tools/run_tests.py的所选算子正确性与core benchmark，保存原始结果和结构化摘要。

测试与基准文件保持上游原样，不启用quick模式。测试收集错误、没有实际用例、缺少有效计时均不能判为通过。跳过项保留说明；通过本步仍需按变更范围完成关联回归、CI与代码评审。配置和解释见[上游验收](../skills/hgkda/references/upstream-validation.md)。

## 三层算子验收的入口

| 层次 | 执行入口与文件 | 验证内容 |
| --- | --- | --- |
| KGS 完整评测 | runtime evaluate；runtime/evaluate.py → adapters/kgs.py → 外部 KGS 服务 | 固定合同的正确性、完整调用性能与用例覆盖 |
| HG-KDA 补充检查 | 同一 evaluate 动作调用 operators/<算子>/checks/，配置绑定检查器 SHA | PyTorch／CANN 双参考、布局、特殊值、别名和算子所需运行语义 |
| FlagGems 原生测试 | runtime upstream-test；runtime/upstream_test.py → 外部 FlagGems tests 与 benchmark | 上游接口、设备分发、原生正确性与原生计时兼容 |

validation.py 用于解释补测结果，不是全部补充测试的执行入口。三层证据绑定同一接入源码 SHA；修改源码后重新验证，原生 benchmark 的数值不替代 KGS 性能分数。测试 HG-KDA 软件本身的 development/tests 不属于这三层算子验收。
