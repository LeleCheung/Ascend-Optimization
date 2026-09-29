# KG v6.7.0 / KGS v6.5.0 H20 发布 tag E2E

本次在 NVIDIA H20-3e 物理卡 7 上验收**已发布 tag 对应的最终提交**：KG `v6.7.0@83b449aa2edb3eb5bf69141eeeaf1e388fe8f27b`、KGS `v6.5.0@1ebddd2f3c160a036c20fbed18986613e1992799`、Protocol v6.2。KG 的全新隔离 venv 通过 `pip install -e .` 安装，从依赖元数据核对 `kernelgen-server-client` 实际来自锁定 KGS commit；目标容器 `kernelgen-nvidia-ngc-2504` 内由 `kg server start` 经 Gitee 准备独立、干净的 KGS exact checkout 和 client/server 安装。远端 `/status.server_version=v6.5.0`、`api_version=v6.2`，backend `cuda`、timing `triton`、NCU Profile 可用，启动后单设备 slot 健康空闲。

输入为 KGS 已安装的 Native Catalog `kernelgenbench` / `kernelgenbench_square`。使用 `kg run --mode kernelgen --catalog-name kernelgenbench --definition kernelgenbench_square --eval-server http://127.0.0.1:19612 --n-parallel 1 --n-epoch 1 --max-round 1 --min-rounds 1 --profile` 后台提交，未跳过测试审核。Runtime 为 Claude CLI，实际模型进程使用 `deepseek-v4-flash[1m]`；RunRequest 未显式指定模型。`2026-09-25T15:00:37Z` 启动，`15:40:03Z` 结束，进程退出码 0。

| 核验项 | 结果 |
| --- | --- |
| `kg status --detail` | `SUCCEEDED`；`prepare_catalog`、`review_tests`、`optimize`、`code_review` 四个阶段均 `SUCCEEDED` |
| 测试契约审核 | `ACCEPTED`，无 blocker；1 条覆盖面建议 |
| 单轮候选 Eval / ledger | 36/36 workload `PASSED`；唯一 measured round 的 geo mean 为 `0.9909924×` |
| Profile | NCU `completed`；选取 5 个代表性 workload，包含 metrics 和部分 instruction 报告；分析文件状态为 `completed` |
| 最终独立复验 | `PASSED` / `VERIFIED`，geo mean `1.0088769×`；该数值与搜索轮次的 ledger 值分开记录 |
| 代码审核 | 阶段 `SUCCEEDED`，无 P0 blocker；3 条 P1 可移植性/健壮性建议，未阻塞此次验收 |
| KGS 收尾 | 停止前 scheduler 为 `device_slots=healthy=available=1`、`active=waiting=checking=broken=incidents=0`；临时 KGS 经 `kg server stop` 关闭，本地 `19612` 和远端 `18312` 端口均不再监听 |

原始工作区在本机 Git 忽略目录 `/data/akg_kernel_bench_lite/kernelgen/runs/kg-v670-h20-tag-e2e-w4LmLb/h20-run/`，Run ID 为 `258538e99fbd4283985d7a17a59cd6af`。关键证据是 `.kernelgen/run-request.json`、`.kernelgen/run-process.json`、`stages/review_tests/attempts/01/review.json`、`stages/optimize/work/1R/agent0/.ledger.json`、同目录 `.kernelgen/evals/round-0001/result.json`、`.kernelgen/profile-analysis/round-0001.json`、`.kernelgen/final-verification.json` 和 `stages/code_review/attempts/01/review.json`。重看状态可用该实验目录的 `venv/bin/kg status <上述工作区> --detail`，并设置 `KERNELGEN_CLI_HOME=<实验目录>/cli-home-final`；性能历史用 `kg history <上述工作区> --json`。

启动前旧的 H20 开发快照 KGS 实例已确认空闲并正常停止，避免两个 KGS 同时管理物理卡 7；未停止或修改占用全部八卡的驻留 Python 进程。目标环境 Torch `2.7.0a0+79aa17489c.nv25.04`、Triton `3.2.0`、CUDA `12.9` 在安装前后保持一致。共用卡上的 `0.991×` 与 `1.009×` 均不宜解读为稳定加速或独占性能结论；本次证明的是**最终 tag 的部署、审核、单轮生成、Eval、Profile、最终复验、代码审核和资源回收**链路完整可用。模型在 Profile 指标分析上花费了明显多于实际 GPU Eval 的时间，这是后续缩短单轮验收耗时的候选改进点，不影响本次正确性结论。
