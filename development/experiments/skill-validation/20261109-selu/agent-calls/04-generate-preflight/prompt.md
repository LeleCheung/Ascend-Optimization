继续独立 HG-KDA selu_ 生成验证。本次不提供历史最佳代码，未新增任何候选；前次因轮数耗尽停在读证据。
先读取 skills/hgkda/SKILL.md、starter.py、context-brief.json，再调用工具桥 context，随后立即生成 candidate.py 和 diagnosis.json 并调用 preflight。
已有同输入成对 profiling、实际 IR 与分类知识，不需要再做目录探索或读全量 comparison.json/context.json。若需要 IR，只读 actual-ir 文件的前100行。
实现必须 run(A)，原地返回 A；所有设备计算和搬运严格 Triton，包含非连续输入，不能用 torch.selu 或原 ATen fallback。可用 torch.empty 仅分配。
diagnosis.json 记录证据、机制假设、修改及预期。最多三份不同候选；preflight --source 当前目录candidate.py，Bash timeout600000；失败后读取简要错误并修复，42/42通过后停止。正式 KGS 由宿主审查后执行。
仅在当前工作目录写文件，不读历史最佳或其他实验，不调用其他 agent。不要只分析而不生成，完成至少一次候选预检。

工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/Ascend-Optimization/tools/skill_validation/agent_tools.py --config /data/hanle/ascend-optimization/runs/independent-skill-selu-20261109-v1/config.json