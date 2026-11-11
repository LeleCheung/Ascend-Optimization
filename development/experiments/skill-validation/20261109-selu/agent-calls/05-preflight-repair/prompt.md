继续已有 selu_ 候选，不从头探索。先读取 skills/hgkda/SKILL.md 和当前 candidate.py。已有成对证据与IR，不需要重新阅读大文件。
上一调用已经生成 candidate.py，但尚无任何设备预检。上次静态门禁拒绝 torch.npu.device；它仅用于宿主设备上下文，可改为已导入 torch_npu 的 torch_npu.npu.device，仍须保留设备上下文语义。不要改门禁或工具。
立即调用工具桥 preflight --source ./candidate.py，timeout600000；读取错误后自行修复源码，再预检。最多三份进入设备预检的不同源码，不重复同源码。静态失败不占设备预检名额。42/42通过后停止。
输出 diagnosis.json，说明从同输入 profiling 和实际IR读到了什么、修改机制、验证及未决问题。保留全部计算严格Triton及原地、布局、offset、精度语义，不用ATen/CANN回退，不读取历史最佳或其他实验目录，不调用其他agent。不要仅分析不验证。

工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/Ascend-Optimization/tools/skill_validation/agent_tools.py --config /data/hanle/ascend-optimization/runs/independent-skill-selu-20261109-v1/config.json