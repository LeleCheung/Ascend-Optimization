加载 skills/hgkda/SKILL.md，阅读 task.json、context-brief.json 和当前 performance-feedback.json。

本次是现有严格Triton交付的兼容性恢复，尚不是性能优化。正式评测在当前设备环境报错：生成launcher收到Python整数地址，访问参数Tensor.size()时失败。PyTorch/CANN补充检查已经通过，说明不应同时重写数学公式或分块算法。

请检查当前 candidate.py 的缓存launcher调用，将指针参数按当前launcher接口传入本次调用的真实张量对象。保持设备kernel、标量实参顺序、分块、精度、缓存键、当前stream查询和冷启动只执行一次的语义。不得将整数伪装成张量或依据评测环境走不同计算路径；不得调用其他后端代算。先阅读上下文里的runtime API摘录，不探测宿主设备环境。

只用Edit修改 candidate.py、diagnosis.json。具体改动前通过工具桥decision登记本轮输入SHA、错误证据、兼容性假设、预期与验证。候选必须由你生成，宿主Codex只作审查与调度。修改后调用工具桥preflight；若返回awaiting_source_review，保存诊断后返回，交由宿主审查，不循环重试。不要编写探测脚本，不改合同、工具或知识包。当前仅修复此机制，真实性能恢复在完整结果回来后继续。
