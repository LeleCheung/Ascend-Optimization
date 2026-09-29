# 昇腾使用 NV 代码作只读 reference 的 CLI 验证

一轮真实 `kg run --mode simple_opt` 已在昇腾 10.0.0.9 完成：CLI 为 `SUCCEEDED`，四个 Workflow 调用均完成，ledger 记录 `PASSED 33/33`、搜索 geo mean **1.0566×**，最终独立复验 `PASSED 1.0550×`。但最佳代码与输入的 NV reference 文件 SHA-256 完全相同，因此这证明 reference-code 输入、后台进度、生成评测和最终复验链路可用，**不证明模型获得了新的昇腾特化实现**。本算子在先前跨芯片复测已达 V2 的 0.8 门槛；“不达标后特化”只是此次工作流演示的假设。

## 冻结输入和执行环境

- KG 代码为已合入 [!166](https://gitee.com/BaaiAC/kernelgen/pulls/166) 的 `dev@1fafbb29`，从已同步该提交的 `codex/operator-guides@68620471` 工作区导入；KGS 为 `main@41101fcbdee18e486402c196980dc204b6598b8e` / Protocol v6.2，Gems 为干净 checkout `4772d816bc5d52d37c4718f36adf377d81261f83`。软件 release 元数据未作为协议判断依据。
- 输入是先前从 `tests/test_relu6_.py` 导出的 Gems Definition Catalog，Bundle `sha256:099b6144e10fc497eb77e93dbd4aaa1cd7b9c495a39fb5a4724afcb89963ffef`。NV 最优 `.best_kernel.py` 的 SHA-256 为 `e297ed3a9b268193d7cdff6f0087638a45f15a8114f8148c52e11b6b1ac3a535`，仅通过 `--reference-code-path` 提供只读设计参考；`run-request.json` 中 `seed_code_path=null`，未继承 NV 的性能成绩。
- 单卡临时 KGS 在 `kernelgen-ascend-flagtree060` 容器内只可见物理 NPU 4，backend `npu`、timing `profiler`，监听远端 loopback 23306，通过本地 SSH stdio proxy 24306 访问。原 KGS Debug Job 预约该 slot，未停止原服务。复用目标原解释器，未升级 Torch、Triton、torch_npu 或厂商运行时。
- 使用现有 `env.sh` 配置的 Claude Runtime；RunRequest 没有显式 `model` 值，因此不能从持久事实确认具体模型名。实验显式 `--skip-review`，因为同一测试契约已审核并在昇腾完成 reference-only；它没有跳过目标 baseline、候选 Preflight/Eval 或代码审核。正式对比实验应显式传 `--model` 并记录实际模型。

## CLI 调用和结果

本次单算子调用使用 `--mode simple_opt --definition relu6_ --catalog-path <已导出的Catalog> --reference-code-path <NV最佳代码> --eval-server http://127.0.0.1:24306 --skip-review --min-rounds 1 --max-round 1 --no-profile --workspace <新目录>`。`kg run` 返回 `SUBMITTED` 与后台 PID；期间 `kg status <workspace>`、`kg status <workspace> --detail` 依次显示 `prepare_catalog → review_tests → optimize → code_review`，最终 `SUCCEEDED`、`process_alive=false`、4/4 调用完成。`kg history <workspace> --json` 从 ledger 投影出一轮 `PASSED`，不是从日志关键字推断。

原始工作区为 `/data/akg_kernel_bench_lite/kernelgen/runs/ascend-relu6-refcode-5RVKX7/run`。相邻 `summarize.py` 从 RunRequest、progress、ledger、`optimize_definition_output.json`、最终复验与代码审核报告生成 `result.json` / `result.md`：搜索 1.0566×，最终复验 1.0550×，最终候选与 NV reference 的 SHA 相同；代码审核为 advisory，记录两项 P1 target-capability 提醒，未阻断已通过的实际评测。该结果不是仓库原生 correctness pytest 的独立候选注入验收，也不是最终 FlagGems PR 验收。

优化与复验完成后，活动 KGS operation 登记为空、临时 KGS scheduler 为 1/1 healthy/available，随后正常关闭；预约 Debug Job 以 `SUCCEEDED`、exit 0 结束，原 KGS 恢复 8/8 healthy/available，active/waiting/checking/broken/incidents 均为 0。本次创建的本地代理已停止，原有服务、用户进程与冻结输入证据未清理或改写。
