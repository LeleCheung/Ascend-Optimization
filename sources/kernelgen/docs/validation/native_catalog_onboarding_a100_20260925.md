# ONBOARDING Native Catalog 入口：A100 实测

ONBOARDING 所示的 KGS 已安装 Native Catalog 路径已用真实后台 CLI 验证：`kernelgenbench/kernelgenbench_square` 在 A100 上完成默认 `review_tests → optimize → code_review`，`kg status` 终态 `SUCCEEDED`、4/4 调用完成且进程退出。ledger 一轮 `PASSED 36/36`，搜索 geo mean **1.0173×**，最终独立复验 `PASSED 1.0301×`。这是功能 Smoke，不作为该算子的正式性能排名。

执行时 KG 代码来自已同步 `dev@1fafbb29` 的 `codex/operator-guides@36bfe29a`，KGS 为 `main@41101fcbdee18e486402c196980dc204b6598b8e` / Protocol v6.2；没有根据 release 推断能力。临时 KGS 只监听本机 loopback，backend `cuda`、timing `triton`，使用 `kernelgen-nvidia-cu128` 容器可见的物理 A100 2。KGS `/operator-contract` 确认目标为 v6.2 `native`，定义具有 27 条 correctness 和 9 条 timing workload；未安装 Gems，也没有升级 Torch、Triton 或厂商运行时。

本次在 ONBOARDING 的最短命令上附加 `--mode simple_opt --eval-server http://127.0.0.1:18080 --min-rounds 1 --max-round 1 --no-profile --workspace <新目录>`，默认测试审核没有跳过。原始工作区为 `/data/akg_kernel_bench_lite/kernelgen/runs/native-onboarding-a100-4oI88f/run`；相邻 `summarize.py` 从 RunRequest、progress、ledger、最终输出和代码审核生成 `result.json` / `result.md`，不手抄性能数字。代码审核策略为 advisory，记录两项 P1 提醒而不覆盖真实 Eval 事实。

本次启动的临时 KGS 已正常关闭，预约 Debug Job 在 `cuda:2` 以 `SUCCEEDED`、exit 0 结束；原 KGS 的该 slot 恢复 available、healthy，未新增 incident。原 KGS 同时有另一用户的活动任务，因此不把整个服务写成“8 卡全空闲”，也未停止或修改该任务。模型使用本地 `env.sh` 配置的 Claude Runtime；RunRequest 没有显式模型名，不能据此宣称具体模型版本。
