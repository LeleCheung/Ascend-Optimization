继续已生成候选的最终正确性修复，不重新探索目录或性能方案。
读取 skills/hgkda/SKILL.md、candidate.py 和 source-review-feedback.json。候选2cdcf5ca87132已通过42/42补测，但源码审查要求修复：传入Triton的Tensor指针已经指向视图的第一个元素；非连续路径再次加 x.storage_offset() 会重复偏移。保留递归高维切片所需的相对base，但顶层从相对0开始。
只修复这个正确性问题，随后立即用工具桥 preflight --source ./candidate.py（timeout600000）。已有两份设备预检记录，本次是第三份。42/42通过后停止修改。
然后用Write工具写入合法的 diagnosis.json：证据、机制假设、已有分块/常量/类型/stride修改、编译报错修复、Codex指出的offset问题、预检结果和未决性能。不要使用shell heredoc生成JSON，不要再做目录/源码搜索，不读历史最佳，不调用其他agent。
所有设备计算及搬运保持严格Triton。正式KGS与额外偏移测试由宿主执行，你不需要运行这两项。最终输出一句完成状态。

工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/Ascend-Optimization/tools/skill_validation/agent_tools.py --config /data/hanle/ascend-optimization/runs/independent-skill-selu-20261109-v1/config.json