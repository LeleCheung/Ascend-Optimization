# 统一优化入口：源码组织与默认值回归

本记录只覆盖 `codex/operator-optimization-entry` 的源码重组和默认值，不表示 pytest Definition Workflow、Gems kernelgen-dev 配套或 H20 E2E 已完成。该后续目标仍需独立实现和真实验证。

基线为 KG `011a8406`，测试使用 KGS `87bc514` 主工作区中的客户端模块；当前发布锁定仍为运行时代码等价的 `376ae336`。未发布或移动 tag。测试前确认 KG 从 `/tmp/kg-operator-import-WWeWzc/kernelgen` 链接导入，链接实际指向本 feature worktree，未将主目录 editable install 当作 feature 验证。

## 验证结果

- 定向入口/模式/单 Coder/KernelGen/CLI 回归：168 passed、1 skipped（宿主无 Torch 的用例）。
- 扩展到本次修改涉及的 host 测试及共享参数、HTTP service、Batch、进度投影：499 passed、30 failed、1 skipped。
- 对上述失败所在的六个测试文件，在未修改的主线另跑：79 passed、同样 30 failed。比较 JUnit 后，失败测试 ID 完全一致；差异消息仅为临时目录编号与本次输入模型类名改变。未把已有失败记为通过，也未混入这些旧夹具/版本断言的修复。

现有失败集中于最终复验缺失导致 NEEDS_RETEST 的旧 Coder/选择夹具、旧 launcher 错误预期，以及 HTTP config 中硬编码 v6.5.0 的断言。原始 JUnit 保存在主工作区 `runs/operator-entry-20260924/baseline.xml` 和 `feature.xml`。

新增检查覆盖：Python/CLI 默认均为 Gems + kernelgen + 1 Coder + 1 epoch + max_round=10；Profile 沿用 KernelGen 默认开启；显式 SimpleOpt/本地 Native 输入仍可用；显式历史参数序列化后不变；Batch 层级覆盖保持原规则；旧 package-level 导入是同一类对象的别名而非第二份实现。

已存在的计划名、stage scope、结果文件、取消 generation 与恢复语义由原测试继续覆盖。多 Coder 测试改为显式传入原有的 2/3 个 Coder，不因新默认值改变其测试主题。

## 测试选择事故

首次扩大范围时，文件列表误包含了手动入口 `tests/run_e2e_full.py`。导入它触发 `/tmp/kernelgen_e2e` 重建逻辑，随后因旧算子 `flaggems_rsqrt` 不存在而退出，没有启动模型或远端计算。该旧脚本会删除同名旧目录，无法从本轮证据确认此前是否有内容。后续选择严格限制为 `test_*.py`；没有把这次手动脚本初始化作为 E2E 成功证据。
