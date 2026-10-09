# narrow_copy 对照实验

起点是提供的同一份 seed。第一轮先原样提交 seed，完成正式 preflight 和 eval，之后才修改候选。两轮都使用完整 FlagGems workload 和 PyTorch 原生 baseline，不得用调试计时代替 `eval_round`。

读取 `eval_round` 返回的 `profile_required`。只有返回 `true` 且能指导下一轮时，才在 `finalize_round` 前调用 `kernel-profile-analyzer`，完成 `get_profile_context`、代表性 `profile_workloads` 和 `record_profile_analysis`，确认 `recorded=true` 后再改第二轮代码。返回 `false` 时直接按正常流程继续；不要臆造 profile 证据。profile 采集时间只是诊断，不是正式加速比。

第二轮一次只验证一个有明确依据的改动。每轮保存候选源码、正确性和逐 case walltime；最终以独立复验为准。不要修改共享 FlagGems、KGS 19650 或其他运行任务。
