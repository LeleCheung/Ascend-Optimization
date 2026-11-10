加载并遵循 skills/hgkda/SKILL.md，任务见 task.json。只在当前工作目录读取和写入，不读取历史最佳源码或其他实验目录，不使用其他agent。
先调用工具桥context，按需读取context-brief.json与实际IR。按task.json中的算子和合同，从starter.py或当前candidate.py生成并优化candidate.py，公共入口run(A)，其余辅助函数单下划线命名。
所有设备计算、布局处理和回退严格Triton，CANN仅为参考。原始starter可能含非Triton后端，必须由你自行改写，不得保留。Codex不直接修补候选，你负责全部候选实现及编译/语义修复。
依据已落盘workflow与知识，输出diagnosis.json：证据、机制假设、修改、预期、验证与未知项。缺失采集保留未知，不虚构IR或性能。
完成后调用工具桥preflight --source <当前目录candidate.py>，Bash timeout设置600000。按失败信息修改新源码，不重复同源码测试。达到task.json的预检要求后返回，供宿主独立审查与正式KGS评测；这只是模型调用边界，不是优化结束。
若已有performance-feedback.json或next-performance-feedback.json，优先检查其源码身份与逐case损失，性能未达标继续修改。不得声称预检通过即达到性能目标。正式结果由宿主反馈，不能自行提交或捏造。
工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/.queued/log10-method-transfer-20261110/package-v1/tools/skill_validation/agent_tools.py --config /data/hanle/ascend-optimization/runs/log10-method-transfer-20261110-development-v1/config.json