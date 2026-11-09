加载并遵循 skills/hgkda/SKILL.md，任务见 task.json。仅在当前工作目录读取和写入，不读取历史最佳源码或其他实验目录，不使用其他 agent。
先调用工具桥的 context 动作读取同输入 CANN 成对 profiling 及实际 IR 摘要；优先读取 context-brief.json 和 actual-ir 内的文件，每次读取不超过100行，避免读取完整大 JSON。
从 starter.py 优化 selu_，输出 candidate.py 及 diagnosis.json。所有设备计算和搬运必须 Triton，禁止原生/CANN/ATen fallback，包括非连续输入。
诊断需包含证据、机制、具体修改、预期变化和未决问题。可调整连续分块、精度、非连续寻址和启动组织。
调用工具桥 preflight --source <当前目录candidate.py>，Bash timeout 设置600000。根据失败信息修复，最多三份不同候选，不重复同源码测试。全部42项预检通过后停止。
先不要提交正式评测或虚构成绩；宿主逐路径审查后将调用 KGS，并反馈正式结果。最终简短说明候选文件及预检状态。
工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/Ascend-Optimization/tools/skill_validation/agent_tools.py --config /data/hanle/ascend-optimization/runs/independent-skill-selu-20261109-v1/config.json