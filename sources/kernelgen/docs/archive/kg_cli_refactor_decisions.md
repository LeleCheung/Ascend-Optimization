# KG CLI 小重构决策（2026-09-06）

历史决策快照：下文“没有实现”和版本边界仅描述当次提交。当前行为见 [CLI 契约](../design/kg_cli.md)，部署见 [远端指南](../operations/deployment/remote_server.md)，不从本记录选择旧 feature 或 KGS commit。

本轮在用户指定的两个现有 feature 分支中实施，不新增产品能力。KG 为 `feat/runtime-observability-control`，KGS 为 `feat/cancellable-operations`；发布版本暂沿用开发基线，部署快照使用 KG 锁定清单中的精确 KGS commit，不移动已发布 tag。

| 主题 | 落地方式 | 保留的边界 |
|---|---|---|
| 优化参数 | KG、Python launcher、YAML 共用 `framework/run_options.py`；删除任意 argv 透传和第二套 YAML 参数模型 | 两种模式各自有效参数、launcher 复用、KG 生命周期参数 |
| 状态 | status/history 共用 ledger 投影；progress 不持久化轮次和性能缓存 | ledger 性能事实、实时 scope/epoch、进程身份、稳定 JSON Schema |
| 续跑 | owner 判断、远端 operation 核实、ledger 恢复、generation 清除、提交统一在 workspace 提交锁内 | workspace 独占、generation 防旧请求竞态、不自动升级 campaign |
| 取消 | 手工、Batch 回收、Ctrl+C 共用服务；SIGINT 通知线程，不中断模型输出；不确定的远端请求保留登记 | 完整模型输出/session、安全点、active operation 持久性、KGS slot 强探针 |
| Server 生命周期 | 管理实现迁入 KGS；KG 只发送 Git bootstrap 并调用目标 checkout 的管理命令；PID 判断共用 KGS 纯函数 | SSH/container 与本地进程边界、loopback、精确 clean checkout、PID reuse 防护 |
| 安装 | 删除离线模式、安装许可开关和 remote_deployed；根据实际导入结果准备轻量依赖 | 不安装或替换 Torch/Triton/厂商运行时、安装前后版本证据 |
| API/client | Preflight/Eval 共用执行与错误映射；三个客户端调用不再为可选 operation ID 复制路径 | 独立 response Schema、Profile 专属流程、单次执行 POST |
| Bundle | 删除 HTTP 层重复 SHA-256 计算 | 客户端快速校验、Store 权威安全校验、原子安装；evaluation_binding=false |

H800 验收发现并修复了共享参数的 tolerance mode 默认值回归，增加 launcher 默认值进入 Workflow contract 的测试。远端生命周期请求使用实例已记录的 commit，避免新锁定清单影响旧实例的 status/logs/stop。没有增加旧本地 client 的运行时兼容 shim，安装说明与锁定清单统一到配套快照。

本轮没有实现 `--operator-dir`、Bundle evaluation binding、TTL/lease 或 Web；这些仍是明确的规划边界。Agent worker lease、KGS 请求线程和 device slot 继续是三种独立资源。
