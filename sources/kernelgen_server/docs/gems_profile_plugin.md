# Gems profiling：显式 pytest 插件

KGS runner 通过 `pytest.main(pytest_args, plugins=[ProfilePlugin(backend, case_id)])` 显式注入采集能力。Gems benchmark conftest 注册一个 first-result hookspec：`pytest_flaggems_profile_scope(backend, case_id)`，返回包住候选迭代和最后设备同步的 context manager。这里的 backend 是 Gems vendor 名称，KGS 插件负责与目标 backend/case_id 比对。

Gems 仍拥有 case 枚举、输入构造、warmup、iterations 和同步；KGS 仍拥有编译器准备、厂商采集、进程超时/取消和报告。编译器准备在 pytest 导入候选前执行。pytest、候选与采集 context 在同一个独立 runner 进程中，不把 pytest 子进程当成 native runner 的 invoke 回调。

Gems adapter 为 inspect、Preflight、Eval、Profile 的子进程按已确定的 KGS backend 设置 `GEMS_VENDOR`，不要求用户再次指定厂商。`runtime.flaggems.flaggems_vendor()` 是唯一名称映射：cuda→nvidia、npu→ascend、musa→mthreads、mlu→cambricon，其余名称保持一致；Profile 插件复用它校验 vendor。该过程只是翻译已有 backend，不再次探测硬件，不修改 KGS 的全局环境；继承的错误 `GEMS_VENDOR` 不能覆盖 Server 的 backend。

本设计删除环境变量模块发现、动态 import、`__main__` 约定、`_ACTIVE_HOOK` 和全局函数替换，不保留旧桥接的兼容分支。Gems 不导入 KGS；可独立运行 `--profile-only`，没有提供 context 的插件时只做普通 candidate-only 重放。

KGS 每次 run 创建一个插件实例，只有正确的 backend/case 完成 capture 后才计数。pytest 退出 0 且该实例恰好完成一次 capture 才写 completion marker；未调用、错误 case/backend、重复 capture、候选异常及 pytest 非零退出均不能伪装成成功。异常时 context 的 finally 继续负责关闭采集。不同 runner 调用不共享插件计数；仍保持每次请求独立进程，不将 pytest 改为常驻多 session 服务。

`compatibility.yaml` 固定包含该 hookspec 的 Gems commit。HTTP Protocol 和 `/profile` 请求/响应没有变化。旧八平台报告属于旧桥接版本的测试记录，不能自动算作新插件接线的验证；新增验证在同分支另行记录。

## 2026-09-23 初次验证

配套为 KGS `a8a7f4c9f3d346617d60b7d57f658236ee81375d` / Gems `79ab71d17b7f5437bdd64185cf570c12c36bf630` / Protocol v6.2。KGS 相关回归 101 passed（含 runner 23 项），Gems 相关回归 73 passed，KG 配套回归 107 passed；均核对实际导入对应 feature worktree。Gems 的插件测试使用真实 pytest plugin manager，覆盖有/无 provider、warmup 与采集顺序、候选失败时的清理；KGS 覆盖缺失/重复 capture、错误 backend/case、失败/旧 marker、独立实例及编译器准备顺序。

真机验证使用本地 A100 的 `kernelgen-nvidia-cu128` 容器卡 7，以及 inventory 中昇腾 10.0.0.9 的卡 7。两者均为独立临时 KGS（A100 loopback 8022；昇腾远端 loopback 24216，经 SSH stdio proxy 25216），1 个设备 slot、4 个请求线程。远端源码通过 Gitee 获取。没有修改 Torch、Triton、厂商运行时或旧 E2E workspace。

| 检查 | A100 | 昇腾 |
| --- | --- | --- |
| 真实 Gems `/profile`，negative 的一个小型 float32 case | completed，NCU report/指标中含 negative_kernel | completed，msprof 报告/摘要中含 negative_kernel |
| 候选在 warmup 后的 capture 内故意抛错 | failed，错误日志保留预期异常 | failed，拒绝缺失的 completion marker |
| 请求结束后 scheduler | active=waiting=broken=0，available=1 | 同左 |
| 普通 Preflight / Eval 回归 | PASSED 15 / PASSED 33 | 本轮不重复全量 Eval |
| 独立 pytest，无插件的 profile-only 重放 | 通过；有插件的直接 runner 也通过且生成 marker | 通过真实 KGS 插件路径验证 |

本轮证明公共插件接线和失败边界，不将其扩展为再次完成八平台、多 epoch 或性能优化实验。旧八平台功能结果仍见原验证报告。测试证据保存在 KG 工作区 `runs/gems-profile-plugin-20260923/`，包括新配套 commit、请求/响应、调度状态、厂商产物及临时实例收尾记录。旧昇腾实例停止时的一次大体积全量证据下载超时，部分下载单独保留；原远端文件及已有本地 Profile 产物没有删除，不将不完整归档标为成功。

Gems 原 PR 合入后，后续开发已迁移到基于最新 master 的 `codex/pytest-profile-plugin`，配套清单固定该分支的 exact commit。独立的八平台 API 回归及环境条件见 [fresh-master 真机回归](gems_profile_plugin_matrix.md)，不将上面的历史结果替换为新版本结果。
