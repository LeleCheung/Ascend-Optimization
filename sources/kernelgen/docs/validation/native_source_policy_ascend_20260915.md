# Native 源策略：昇腾单卡验真（2026-09-15）

## 验证范围与组合

本次验证新增 `support_fp64` 源策略在真实昇腾 NPU 上的解析、reference 精度分支及条件 workload，不是模型生成优化实验。KG v6.4.0 开发分支 `fix/advisory-code-review@da48958b` / KGS v6.3.4 开发分支 `feat/native-source-policy@9af6ef86d32af2379bc5abf701e4d37293064537` / Protocol v6.2；这不是新 release。功能可用性按 `/status.capabilities.native_source_policy` 检查，不按软件版本猜测。

临时 KGS 通过 Gitee 的 `test-native-source-policy-20260915` 分支取得 exact clean checkout，KG 开发工作区的 lock 仅在创建测试实例期间临时指向该提交，随后恢复原值 `49fcf4ec`。没有更新正式发布组合或移动 tag。

## 部署和资源

按用户授权，先确认原实例 `kt2-stage2-ascend-EdHzTP` 空闲，验证远端 PID/start identity 后停止，更新已经失效的 SSH 地址，再将设备配置改为物理 0～6、7 workers。原 KGS 保持 `00dfc8692a6927d052b0c6da238cdde7908363c4`，不升级既有实验环境。现行 CLI 的跨 KG release 检查拒绝直接启动旧实例，本次使用现有受管启动 primitive 恢复；保留 endpoint/device 锁、exact checkout 校验、readiness 和 PID 登记，没有修改旧实例的 release 字段。复核 7 个 slot 全部健康、空闲。

临时实例 `source-policy-npu-OoqmNZ` 使用容器 `codex_fib_ascend_20260821` 的物理 7 号卡，`ASCEND_RT_VISIBLE_DEVICES=7`，映射为逻辑 `npu:0`，1 worker/1 slot，timing 为 `profiler`。远端仅监听 `127.0.0.1:22116`，本机由 SSH stdio proxy 提供 `127.0.0.1:23116`。没有重启机器、重置卡或停止其他服务。

独立解释器为 `/home/secure/xuyao/kernelgen_e2e_ascend_20260821/deployments/source-policy-npu-OoqmNZ/venv/bin/python`。Debug Job 确认导入本次远端 checkout，Torch `2.9.0+cpu`、torch_npu `2.9.0.post2`、Triton `3.5.1`（FlagTree `0.6.0+ascend3.5` 提供），CANN `9.0.0`，driver `25.2.0`。复用系统厂商运行时，未安装或替换上述核心包；新虚拟环境安装本次 KGS 及轻量依赖 safetensors `0.8.0`，其他管理快照中的依赖版本不变。Triton 模块可用，但其 distribution 名是 FlagTree，不把 `metadata.version('triton')` 查不到解释成模块缺失。

## 结果

| 检查 | 结果 |
| --- | --- |
| `/status` 源策略 | vendor=`ascend`，fp64=false，bf16=true，int64=true |
| 单卡 reference 分支 | oracle 断言 fp64=false、参考 Tensor 为 NPU float32，通过 |
| 条件 fp64 workload | correctness 与 timing 在分配前 SKIP，保留原因；其余测例 PASSED |
| Preflight | PASSED，并返回条件 timing 的 SKIP 原因 |
| 故意抛出的执行异常 | RUNTIME_ERROR，不误报 SKIP 或触发 broken slot |
| 两请求并发 | 两次 PASSED；52 次采样中 max_active=1、max_waiting=1、broken=0 |
| 真实抽取的 `fix` | Bundle 上传、contract、inspect、Preflight、Eval 均通过；18 correctness + 15 timing，33/33 PASSED |
| 全部 correctness SKIP | 返回 ALL_SKIP，继续适用的 benchmark，保留逐测例 timing，不发布 headline |
| 不依赖 Gems 导入 | Debug Job 禁止导入 `flag_gems` 后源策略仍可读取，确认进程未导入 Gems；容器本身已有 Gems，因此不声称做了卸载 Gems 的环境测试 |

`fix` 来自此前真实 Codex 抽取和审核的 `runs/kernel_todo_v2/source-policy-codex-qhGxfy/fix/catalog/`，本次没有修改或精简该 Catalog。其 correctness reference 使用 `source_flag("support_fp64")` 选择精度；6 个 bf16 条件测例在昇腾全部适用。候选是 `torch.fix` reference-as-solution，用于验证评测链路，不是高性能 Triton 实现，也没有启动新的 Coder 或重跑抽取模型。原 timing 清单仍来自冻结的采集结果，本次不重新执行源 pytest list-case。

主验证结束时临时实例 `active=waiting=checking=broken=incidents=0`、`healthy=available=1`。本次通过不代表所有历史失败算子已恢复，也不代表其他厂商已实测；保留源 pytest 容差与后续优化质量的独立审核边界。

补充验证结束后已通过 `kg server stop` 正常停止临时实例，保留其 checkout、虚拟环境、日志和请求证据；删除本次 Gitee/local `test-native-source-policy-20260915` 临时分支，开发提交仍保留在 `feat/native-source-policy`。原 0～6 号卡实例继续运行，没有把卡 7 自动加回。原始 Catalog、旧实验 workspace 和主工作区其他人的未提交内容保持不动。

## 证据与管理入口

本地证据根目录为 `/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/source-policy-npu-OoqmNZ/`，不随仓库分发。包含 `validate.py`、请求/响应 JSON、原配置快照、设备状态、Debug Job 环境信息以及本次临时实例 CLI 状态。原服务 CLI home 未迁移，查看命令为：

```bash
KERNELGEN_CLI_HOME=/data/akg_kernel_bench_lite/kernelgen/runs/kernel_todo_v2/stage2_kg631_20260913_EdHzTP/cli-state kg server status kt2-stage2-ascend-EdHzTP --json
```

部署方法遵循 [远端 KGS 部署](../operations/deployment/remote_server.md)；设计见 [Catalog 抽取审核](../design/workflows/catalog_extract_review.md)。原服务仍是旧 KGS，不因旁边的临时实例验真而获得新的源策略 capability。
