# 新实验的 Gems 来源策略

只自动跟踪 Gems，不取消 `deployment/kgs.lock.yaml` 中的 KGS exact commit。Gems 的默认 repository/branch 来自配套 KGS `compatibility.yaml`：官方 `flagos-ai/FlagGems`、`kernelgen-dev`、`revision_policy: branch`。不在 KG 里复制分支常量，也不引入第二套更新器。

## 准备边界

首次 `kg server start <name> --install-gems` 或 `kg server install-flaggems <name>` 解析默认分支 HEAD；保存实际 commit、分支和独立 checkout 路径。配置为跟踪默认分支的实例，后续 `kg server start <name>` 会再次检查 HEAD。没有配置 Gems 的实例不因一次普通 start 就自动安装 Gems。

相同 commit 不重复安装。分支变更时，只有已停止实例能够准备新 checkout；旧 checkout 保留。如果服务还在运行，操作明确失败并保留原实例，不以 scheduler 短暂 idle 作为“没有实验”的依据，不擅自停掉服务。网络查询失败也不悄悄退回旧版本并宣称已更新。

这发生在实验前的服务准备流程，**不是每次 `kg run` 隐式重启共享 KGS，也不是每次 Eval/Profile 更新代码**。新实验开始前先完成 `kg server start` 的来源检查；需要换版本时在旧实验结束后 stop/start。Batch 或多机实验应复用同一个实际 commit，而非逐算子重新取 HEAD。

## 续跑与显式选择

实例、run 的既有快照和 ledger 继续记录实际使用的 commit。`kg resume` 不调用上述分支更新流程。需要恢复原依赖时，先停止对应服务，执行 `kg server install-flaggems <name> --revision <记录的完整commit>`，再 start；完整 commit 或非默认分支的显式选择不被普通 start 自动切回默认分支。

旧 exact-policy KGS 清单保持原行为，避免只更新 KG 就改变旧实例。版本选择以清单、实例和运行记录各自的职责为准：清单给出来源策略，实例保存实际安装结果，实验保存使用快照；Gems 更新不自动推进 KGS commit。
