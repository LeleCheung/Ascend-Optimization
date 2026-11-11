加载并遵循 skills/hgkda/SKILL.md，任务见 task.json。只在当前工作目录读取和写入，不读取历史最佳源码或其他实验目录，不使用其他agent。
宿主已按当前源码与本次固定方法刷新上下文，先读取context-brief.json、contract.json和knowledge.json，并核对源码身份。工具桥context可重新整理证据；用法已在task.json完整给出，不浏览工具实现、help或workspace外的方法清单，这些由宿主校验。已有候选时直接围绕失败反馈修改，不重复探索环境。按task.json中的算子和合同，从starter.py或当前candidate.py生成并优化candidate.py，公共入口run(A)，其余辅助函数单下划线命名。
所有设备计算、布局处理和回退严格Triton，CANN仅为参考。原始starter可能含非Triton后端，必须由你自行改写，不得保留。Codex不直接修补候选，你负责全部候选实现及编译/语义修复。
依据已落盘workflow与知识，输出diagnosis.json：证据、机制假设、修改、预期、验证与未知项。缺失采集保留未知，不虚构IR或性能。
本任务首先验证历史方法能否迁移。对知识中已有成功结构，先核对当前候选是否真正落实了适用规模、载荷大小、循环特化和启动路径；存在明显缺项时优先补齐并验证。不同几何配置的失败不能否定尚未实现的历史方法，也不能仅凭不同dtype耗时相近就断言通用数学已标量化。已有编译或语义失败时先完成最小修复，不同时引入新优化机制。
diagnosis.json已重置为本轮短空模板，旧诊断由宿主保存，不要尝试恢复或逐项修订历史长文档。只填本轮必要字段与证据摘要，控制在约2000个汉字内；不复制完整历史或整段IR。源码注释说明当前实现即可，不持续追加已被取代的长篇实验叙述。已有文件优先小范围Edit；单次Write保持简短。优先完成候选和短诊断并立即调用预检，避免在文档修订上耗尽轮次。读取IR先根据kernel符号、循环、拷贝及缓冲分配定位必要片段，不连续读取整份大文件。Bash权限拒绝后不换解释器重复尝试；不探测宿主numpy等环境，设备测试只走工具桥。
已有正确候选时，本次只实现一个可独立验证的性能机制，不同时重写所有布局与启动路径。优先用Edit落盘小改动，再完善同一机制并交接预检；下一次根据完整性能反馈选择后续机制。
运行时接口的可读摘录路径见context-brief.json的runtime_api.path；优先读取摘录与当前反馈，已有验证且本次未改变的ABI不反复扫描。先落盘最小必要修订，避免把全部轮次消耗在重复读取和诊断说明上。
只编写candidate.py和diagnosis.json，写入或修改使用Write／Edit工具，不使用Bash的heredoc、cat、printf等写文件命令。长实现拆为多次小修改，每次处理一个机制，避免单次重写超过模型输出上限；已有正确布局路径时优先保留它。candidate.py只包含算子实现，不混入输入构造、PyTorch对照、测试或探测脚本。不要预检starter.py，不编写或预检probe脚本；设备环境由工具桥管理，不能自行探测宿主venv或容器。
完成后调用工具桥preflight --source <当前目录candidate.py>，在Bash工具的timeout参数设置600000，不在命令前加timeout包装器。若返回awaiting_source_review，保持当前候选不变并结束本次调用，等待外部只读审查；不要为绕过审查另写源码或反复提交。按失败信息修改新源码，不重复同源码测试。达到task.json的预检要求后返回，供宿主正式KGS评测；这只是模型调用边界，不是优化结束。新写的注释与诊断使用专业中文。
若已有performance-feedback.json或next-performance-feedback.json，优先检查其源码身份与逐case损失，性能未达标继续修改。不得声称预检通过即达到性能目标。正式结果由宿主反馈，不能自行提交或捏造。
若绑定当前源码的正式结果已达任务阈值且全部所需补测通过，保留candidate.py原字节，仅在diagnosis.json记录结论并结束；不为追求更高分无条件再改源码。
工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v36/tools/skill_validation/agent_tools.py --config /data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v2/workspace/agent-config.json