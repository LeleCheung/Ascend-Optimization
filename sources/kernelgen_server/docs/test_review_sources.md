# 测试源码审核证据

`/status.capabilities.operator_contract.test_sources=true` 表示 `/operator-contract` 支持请求字段 `include_test_sources=true`。SDK 使用 `get_operator_contract(request, server, include_test_sources=True)`；默认请求和返回保持原行为，不自动附带源码。

返回的 `test_sources` 含 `framework_revision` 和 `files`（相对路径与 UTF-8 文本）。绑定仍只接受已安装 Catalog 名称或已上传 Bundle，不接受客户端提供任意文件路径。Gems 复用 Adapter 的 checkout、上传来源、suite 和摘要核对，导出原正确性/benchmark 文件、常用测试 helper、flag_gems.testing 源码；Native 使用已有契约中的 reference/workload，并导出 oracle 同目录的 Python 源码。不会执行 pytest、oracle、候选或设备计算，不申请设备 slot，也不表示审核或 readiness 已通过。

总文本上限 8 MiB、文件数上限 128；越界、链接、非 UTF-8 或超过上限明确失败，不返回截断后声称完整的源码。数据归档、二进制和任意第三方依赖不自动导出；KG Reviewer 对缺失的必要依赖应报告阻塞，不猜测其行为。

响应由当前目标绑定产生，不能用其他 checkout 的同名文件替代。KG 应保存并校验导出的证据，同时区分语义审核与来源核对；显式跳过审核不能跳过候选 Preflight、正确性、Benchmark 或设备隔离。
