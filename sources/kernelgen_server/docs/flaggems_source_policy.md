# Gems 分支更新与运行快照

KGS 的 `compatibility.yaml` 为 Gems 声明官方仓库、`kernelgen-dev` 分支和 `revision_policy: branch`，不再重复保存 Gems 的固定推荐 commit。这不改变 KGS 自身的版本选择：KG 的 `deployment/kgs.lock.yaml` 仍锁定 KGS exact commit，Protocol 仍由运行中的 `/status.api_version` 判断。

配套 KG 在新实验的服务准备阶段解析 Gems 分支 HEAD 一次，安装到按实际 commit 区分的独立 checkout，并把实际 commit 写入实例记录。运行中的 KGS 只使用该 checkout；不在 Eval、Profile 或设备 worker 内 `git pull`，不覆盖旧 checkout，不升级 Torch/Triton/厂商运行时。

服务仍在运行时，可以确认当前 Gems 已是最新版本，但不得替换它。如果分支出现新提交，准备操作应报出需要停止旧实例后更新，而不是中断已有实验。`kg resume` 沿用原快照；需要恢复旧环境时显式选择记录中的完整 Gems commit，再启动实例。完整 commit 选择不自动跟随分支。

该声明需要配套 KG 支持 branch policy。旧 KGS 的 exact-policy 声明和已有固定实例仍由 KG 按原规则读取，不用 Gems 分支更新推导 KGS release 或协议兼容性。多机同一实验的编排者应复用一次解析得到的 commit，避免在依次准备机器时选择不同的分支 HEAD。
