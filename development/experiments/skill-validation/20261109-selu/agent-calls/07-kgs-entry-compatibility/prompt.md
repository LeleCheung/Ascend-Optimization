完成KGS入口兼容修复。读取 skills/hgkda/SKILL.md、candidate.py 和 kgs-entry-feedback.json。
候选f3502fad通过42/42 PyTorch补测、额外offset 9/9，但正式KGS在执行前报告 conflicting public symbol selu_。它同时发现候选的 run 与 selu_ 两个公共符号。
只将Python入口辅助函数 selu_ 重命名为 _selu_entry，并让 run 调用 _selu_entry。不要改变任何Triton kernel、常量、分块或数学，不重新探索文件。
用工具桥 preflight --source ./candidate.py，timeout600000。本次额外预检因KGS入口兼容性缺陷而显式授权，前面原件保留。
42/42通过后，用Write更新diagnosis.json的源码SHA与KGS接口修复记录，然后停止。不提交正式评测、不再改任何候选、不读历史最佳、不调用其他agent。

工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/Ascend-Optimization/tools/skill_validation/agent_tools.py --config /data/hanle/ascend-optimization/runs/independent-skill-selu-20261109-v1/config-entry-repair.json