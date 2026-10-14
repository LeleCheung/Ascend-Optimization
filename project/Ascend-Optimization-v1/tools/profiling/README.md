# Ascend Profiling Agent

2026-10-14状态：矩阵乘和amin新候选已完成完整闭环与两次最终复验。logaddexp2历史无profiler/有profiler为1.061×/1.267×；历史方法workflow无人值守复现两次1.269×/1.261×，均36/36、最终SHA与历史版本一致。见[阶段交付](../../operators/一阶段交付-20261014.md)、[历史方法](历史复现知识-20261014.md)与[算子入口](../../operators/logaddexp2/README.md)。

“独立workflow”表示把此前Codex参与的诊断整理与有效方法固化，让服务器自行执行，不要求重新发明优化算法。KG两组标签表示运行开关；复现历史候选与新增优化分别记账。约1.267×是本次复现目标，未采纳新候选时也可完成历史方法交付。

已有 `ascend_profiling_workflow.py` 校验并分析 Eval、真机 msprof、模拟器与原始附件。新增 `optimization_campaign.py` 将诊断反馈给 CC+DeepSeek，再完成候选生成、完整复验、最佳选择与两次最终复验。

每轮从当前最佳版本选择最慢的两个不同形状/dtype 用例，采集 PipeUtilization；向 CC 提供逐 case 延迟、语义工作量、流水指标、失败历史和 910B 知识。CC 仅返回完整源码与优化理由，设备调用由控制器统一执行。只接收相同合同的完整通过结果，超过当前最佳 0.5% 才替换；未提速和失败代码也保留。

若存在`operators/<算子>/优化知识.md`，会与公共910B知识一起输入并保存为批次`knowledge.md`，配置SHA覆盖合并后的全文；恢复时知识改变则停止。logaddexp2的专用知识覆盖稳定公式、NaN/Inf、类型提升、AIV流水与分块覆盖，未包含另外两组KG的候选和成绩。

`--plain-json`绕过CC结构化工具，仅接收最终JSON文本；格式仍由源码AST和完整设备评测校验。`run = 已定义函数`是合法KGS入口，会保留原文；若只提供`logaddexp2`公开实现则记录适配后的源码补入别名。模型生成占位符或不可解析文本仍会失败，不把它算作有效候选。

新优化批次可设置`--thinking-budget 8192 --max-output-tokens 16384`，分别传入CC的思考与总输出预算，并传入`--max-turns 1`。本机DeepSeek/CC仍出现仅有思考的额度耗尽续写，不能认为预算参数已解决所有空转；20分钟超时仍负责最终停止。预算和`--replay`身份写入批次配置，恢复时须一致。

默认历史方法复现使用`--replay`：完整复测固定源码、采集、保存诊断和提示词、最终完整复验两次，不调用模型重新生成。需要继续优化时，在新批次省略`--replay`。复现脚本的第一个参数可指定新批次，已完成批次再次执行只核验并复用；例如`bash tools/profiling/run-logaddexp2-replay-20261014.sh 20261014f-replay`。

复现脚本生成本批`复现汇报.md`，使用`finalize-agent-campaign.py --replay-only`，只比较本批起点与两次最终复验。提交后HEAD会改变合同指纹，新批次不与提交前的旧基线做严格混合对照；历史四版本表保留原合同。已完成批次可在归档提交后只读核验并复用，运行中批次仍禁止改变HEAD。

当前批次：`20261012-agent`。物理卡7、KGS19655、同卡串行。矩阵乘从v9出发，随后amin从v5出发，各6轮；每轮CC有20分钟超时，完整评测有30分钟超时。不会无限生成候选；一批结束后依据结果决定续跑。

## 运行与查看

在910B宿主机运行：

```bash
bash /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/run-agent-campaign-20261012.sh
```

正式批次已由tmux托管，查看：

```bash
tmux attach -t profiling-agent-20261012
cat /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/profiling-agent-20261012-agent/campaign-state.json
```

日志位于 `/data/hanle/ascend-optimization/runs/20261012-agent/campaign-logs/`。SSH断开不影响服务与tmux。单个CC调用有边界，控制器会依次启动后续轮次，无需人工逐轮下指令。

## 恢复和证据

工作区含 `seed-evaluation/`、`round-XX/profiles/`、`candidate.py`、`proposal.json`、`evaluation/`、`campaign-state.json` 和 `final-evaluation/`。状态使用原子替换。程序通过文件锁防止重复启动；服务异常或仍忙时停止，保留状态和原始文件。

恢复同一批次使用原输出目录，已记录轮次不会重跑；中断的候选若已写好，继续完成其评测。采集请求已发但尚无响应时保留目录，不自动重发。更改候选源码、合同或仓库HEAD后使用新批次。**同一组实验未完成前不要提交或pull仓库**：当前KGS指纹包含总仓库HEAD。

Claude入口使用 `runtime/kg-controller/claude` 包装器，模型参数 `deepseek-flash`，它在内存中读取既有DeepSeek配置，不复制凭据；直接使用 `bin/claude` 且禁用settings会缺少认证，本批前两轮的该配置失败证据保留。模型返回占位符时最多做一次格式修复，原响应保留；超时会结束控制器创建的子进程组。

每次设备提交前检查19655与19657。恢复时发现请求文件存在、结果未知则停止重发，先检查远端任务状态。已经完成的campaign直接返回，不重复最终评测。

后续新对象由 `screen-next-operator.py` 先筛查Excel中低于0.8的短名单，并复测固定master。只有完整正确性通过且当前master仍低于0.8才立项；当前短名单为logaddexp2、acosh、add_relu。筛查在前两个campaign结束后串行运行。

`queue-new-operator-20261012.sh` 确认两个campaign完成且筛查记录已生成，取得物理卡7全局锁后运行 `run-new-operator-closure-20261012.py`。新算子的原生KG两组和独立Agent均从同一固定master、同一模型开始，最多六轮；三种优化版分别完整复验两次，版本表选各自较慢的一次。六轮是上限，错误生成次数、真正评测次数和实际耗时也须报告。

2026-10-13 恢复记录：矩阵乘最终两次为0.908×、0.906×；amin为0.942×、0.936×。新对象logaddexp2固定master为0.231×、36/36通过。KG无profiler优化阶段已完成，但后续代码审核因证据路径登记问题停在WAITING_CODE_REVIEW；导出显式使用`--allow-review-waiting`，核实worker已经退出和优化阶段COMPLETED，原生状态与审核意见原样保留，独立复验继续。候选不因此获得上游审核认可。报告脚本现会先创建日志父目录。队列以campaign完成状态和筛查文件为准，并等待物理卡7全局锁。

## 方法范围

已下载全部对照证据时，可在Windows或Linux离线重建报告，不连接KGS：

```bash
python tools/profiling/finalize-agent-campaign.py --operator matmul_bias_activation --prior-closure master-closure-20261012 --offline
```

独立Agent额外续跑时，使用`--batch 20261012-agent --campaign-batch 20261014d-agent`将本轮Agent和原KG对照生成到原closure；报告注明续跑起点、人工修复与额外成本，不能据此声称相同总预算的消融。

离线模式仍核验源码SHA和完整合同；缺失结果就停止，不发起新评测。原始JSON、模型响应和真机附件保持原内容，重建的是汇总报告。实时队列状态仍需通过SSH核实。

数值Roofline需要同一kernel/range的真实计数、耗时及硬件roof；当前尚未满足，不用逻辑字节伪造roofline。优先依据真机流水、算子语义、源码与编译资源约束做修改；模拟器和IR按具体问题补充。完整测试与两次最终性能复验决定交付结果。
