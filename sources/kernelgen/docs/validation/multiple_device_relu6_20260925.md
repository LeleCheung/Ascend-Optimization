# `relu6_`：同一 Gems pytest 与生成候选的跨芯片复测

本轮通过新 `MultipleDeviceTestWorkflow` 对一个冻结的 Gems Definition 和一个冻结的 A100 生成候选逐目标复测。**海光、摩尔线程、沐曦、昇腾**的 core baseline、Preflight 与原 pytest/benchmark Eval 全部通过；燧原 baseline 通过但候选受网格上限阻断。昆仑芯、天数、平头哥分别停在来源加载、连接和部署环境阶段，不能记录为算子或 dtype 不支持。此页保留未完成项，不能概括为“国产芯片全部通过”。

## 冻结输入与执行边界

- KG `codex/multiple-device-test` 开发分支基于合入统一审核的 `dev@ea59b7ed`；KGS `main@3d00e93c592847b9fc3cf8f03a5345a3466dc7ae`，Protocol v6.2；目标 Gems checkout 均要求官方 `kernelgen-dev@4772d816bc5d52d37c4718f36adf377d81261f83`。没有按 KGS/Gems release 猜能力，每台实际读取 `/status.api_version` 和 capabilities。
- 现场结果保留当时的 KGS `3d00e93`，不回填为后来的版本。KGS [!65](https://gitee.com/BaaiAC/kernelgen_server/pulls/65) 仅修复同级源码 checkout 的公开导出，并以 `main@41101fcbdee18e486402c196980dc204b6598b8e` 合入；当前 KG 锁定清单已指向新提交，但尚未用该提交重跑本页全部目标。
- 本地 Definition 来自 `kg definition` 对 `tests/test_relu6_.py` 的导出，Bundle 为 `sha256:099b6144e10fc497eb77e93dbd4aaa1cd7b9c495a39fb5a4724afcb89963ffef`。候选为 [A100 验收](gems_definition_review_a100_20260925.md) 的 `.best_kernel.py`，SHA-256 `e297ed3a9b268193d7cdff6f0087638a45f15a8114f8148c52e11b6b1ac3a535`；所有目标使用相同代码，未引入按芯片修改、seed 或测试删减。
- 各目标通过本地 SSH stdio HTTP proxy 连接其 loopback KGS。KGS 上传并 inspect Bundle 后先执行候选无关 `/reference`；只有 `PASSED` 才请求 Preflight，只有 Preflight `PASSED` 才请求 Eval。原始 KGS 响应（包括逐 case 失败字段）分别保存在每台的 `reference.json`、`preflight.json` 和 `evaluate.json`，Workflow 汇总不代替原报告。
- 使用现有目标容器解释器和受保护运行时；未安装、升级或替换 Torch、Triton、厂商扩展。新服务使用 exact KGS checkout 与单卡可见性；在旧 KGS 已占有设备的机器，先用其 Debug Job 占住一个 slot，再启动同卡临时新 KGS，结束时先停新 KGS、再释放预约。昇腾 10.0.0.9 的 1 号 NPU 现场无进程，使用独立单卡临时实例；未在其他三台昇腾机器重复测试。

## 结果

以下表格由本地原始目标结果运行 `runs/mdt-relu6-20260925/summarize.py` 生成；`geo mean` 是当前目标短时单卡测量，不能用于不同芯片的性能排名。评测设置为 warmup 10 ms、benchmark 10 ms、1 trial，超时上限 600 秒；全部 case 和 dtype 均来自同一 core pytest 契约。

| 芯片 | 结果 | 截止阶段 | baseline | Preflight | Eval | 正确/总数 | geo mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 昇腾 910B4-1 | PASSED | evaluate | PASSED 15/15 | PASSED | PASSED | 33/33 | 1.0409× |
| 海光 BW | PASSED | evaluate | PASSED 15/15 | PASSED | PASSED | 33/33 | 1.1247× |
| 摩尔线程 MTT S5000 | PASSED | evaluate | PASSED 15/15 | PASSED | PASSED | 33/33 | 1.0295× |
| 沐曦 MetaX C550 | PASSED | evaluate | PASSED 15/15 | PASSED | PASSED | 33/33 | 1.0634× |
| 燧原 ZIXIAOC200 | PREFLIGHT_FAILED | preflight | PASSED 15/15 | RUNTIME_ERROR | 未运行 | — | — |
| 昆仑芯 P800 OAM | ERROR | inspect | 未运行 | 未运行 | 未运行 | — | — |
| 天数 | ERROR | status | 未运行 | 未运行 | 未运行 | — | — |
| 平头哥 PPU | ERROR | status | 未运行 | 未运行 | 未运行 | — | — |

燧原失败发生在首个候选 timing Preflight case，Triton 报 `grid.x` 需要 524,288，而目标上限为 65,535。它不是原生 API/dtype 不支持：同台 `/reference` 已对 FP16、FP32、BF16 各 5 个 case 全部通过。该候选没有进入燧原 Eval；后续特化需改变网格映射，并重新走同一完整契约。

昆仑芯已部署并通过单卡 KGS 强探针，但 `/inspect` 的原 benchmark case-list 导入 Gems 时失败：目标旧 Torch 的 `Library.impl()` 不接受 `allow_override` 参数。参考测试尚未开始，不能把它分类为该算子 API、dtype 或候选代码问题；需要解决目标 Torch/Gems 接口兼容后重新 inspect 与 reference，不得只改 Definition 来源摘要。昆仑芯按单算子、单设备、单请求低并发诊断，未启动 Batch 或强模型生成。

用户确认昆仑芯后续通过更新镜像处理，本轮不将针对旧 Torch 的隔离实验补丁合入或用于当前 Workflow，也不将此目标计入当前发布门禁。旧环境失败仍作为历史证据保留；曾用于隔离诊断的远端 Gems 临时 worktree 已移除，原冻结 checkout 未修改。

天数使用当前 `tests/hosts.md` 中的 JumpServer 地址连接时返回“未发现匹配的资产”，没有建立 KGS 代理。平头哥 SSH 可达，但清单端口的 KGS 已停止，项目指定的专用 Gitee 只读部署 key 路径不存在；根 EXT4 文件系统的既有故障仍需避开。两台在 Workflow 中均记录为 `status` 阶段 `ERROR`，没有伪造 `/reference` 或 Eval 结果。待资产入口与专用 key 明确后应重新在各自目标环境部署匹配 KGS/Gems，并以新 workspace 复测。

2026-09-25 复查时，天数的同一 JumpServer 登录仍返回资产不匹配，未见可替代的当前清单入口。平头哥 `/mnt/workspace` 为非 EXT4 挂载，历史 NAS 测试目录和 `KGS@5c3bb659` checkout 仍在；但它不等于当前锁定的 `41101fc`。对该 checkout 做只读 `ls-remote` 失败，原因是外层 `/root/.ssh/config` 存在无效的 `ssh-rsa` 配置行，且清单指定的专用只读 key 不存在。不得以旧 checkout、其他账号密钥或未锁定代码伪装成新 KGS 验收。

用户允许使用本地 `A100_proxy.sh` 后，已在平头哥非 EXT4 的 `/mnt/workspace/kg-gems-4772-20260925` 从官方 GitHub 分支浅克隆干净的 Gems `kernelgen-dev@4772d816`；代理只作为该 Git 命令的临时配置，通过 SSH stdin 传入，未输出、复制为文件或写入仓库配置。原 KGS/Gems checkout 未覆盖。缺失的轻量依赖仅以 `pip --target` 安装到相邻 NAS 目录：SQLAlchemy 2.1.0 和 typing_extensions 4.16.0；系统 Torch 2.9.0、Triton 3.5.0+git43e6ef4d 未变化。对原 benchmark `--reference-only --level core` 得到 `PASSED 15/15`、无 skip；再用进程级 `GEMS_VENDOR=thead`、`CUDA_PATH=/usr/local/cuda`、`PPU_SDK=/usr/local/PPU_SDK` 和全套 NAS 缓存路径运行原 `tests/test_relu6_.py`，正确性 pytest **18/18 passed**。15 号 PPU 测后仅有 2 MiB 基础占用、利用率 0、无进程，源码 checkout 仍干净。早期未设置 `FLAGGEMS_CACHE_DIR` 的诊断曾把生成源码写到默认根缓存；未清理该共享缓存，正式验证已改用 NAS 路径。

这两项是 **Gems 源 pytest 的独立验真**，不是本 Workflow 在平头哥的 KGS `/reference`、Preflight 或 Eval；上表的平头哥 `status ERROR` 仍是当时真实结果。A100 代理可达 GitHub，但经该代理访问 Gitee 返回 CONNECT 500，直接 Gitee HTTPS 要求认证；清单指定的 KGS 仓库专用只读部署 key 仍不存在。尚未用 Gitee exact commit 准备或启动新 KGS，不能把新 Gems 的通过扩大为平头哥完整跨芯片 Workflow 验收。

## 证据、状态与清理

本机原始目录为 `/data/akg_kernel_bench_lite/kernelgen/runs/mdt-relu6-20260925/`：`targets/<芯片>/plan.json` 绑定候选 SHA、Bundle ID、Gems commit、评测设置和目标；`targets/<芯片>/targets/<芯片>/` 保存目标原始 JSON；天数与平头哥的状态失败记录在 `targets/unavailable/`。`results.json` 和 `results.md` 由 `summarize.py` 从上述每台结果派生并核对身份一致性。A100 对新 Workflow 的独立真实 API 冒烟在相邻的 `runs/gems-definition-review-e2e-86KA2I/multiple-device-a100-smoke/`，顺序和结果为 reference 15/15、Preflight PASSED、Eval 33/33 PASSED。

真实调用及 host 测试均证明 baseline 失败的目标不会进入候选 Preflight/Eval，后一个目标仍可继续；无 KGS 或缺少 capability 会记录为环境错误。取消通过同一 workspace Run Control 的 generation 与活动 operation 登记协作转发。原始现场运行使用早期的 `summary.json` 聚合；后续 Workflow 接口已收敛为返回并保存相同内容的 `WorkflowResult[MultipleDeviceTestOutput]`，最终文件为 `workflow_result.json`。每台 `result.json` 和 KGS 原始响应不变，旧成功目标不被另一台失败覆盖；第一版不自动并行或重用历史目标结果。

## 锁定 KGS 与最终 JSON 补测

2026-09-25 使用 KG `codex/multiple-device-test@67feb0e2`、锁定 KGS `main@41101fcbdee18e486402c196980dc204b6598b8e` 和上述 Gems `4772d816`，在 A100 现有 KGS 中先预约一张空闲卡，再用该卡启动只监听 loopback 的单卡临时 KGS。相同 Bundle 与候选通过 core reference **15/15**、Preflight `PASSED`、Eval **33/33 PASSED**，geo mean 为 **1.0029196823**。`runs/mdt-json-e2e-Dh8mnM/workflow/workflow_result.json` 的 `state=SUCCEEDED`、`completed_targets=passed_targets=1`，使用 `MultipleDeviceTestWorkflow.OutputModel.model_validate_json` 重新解析成功；同目录保留 `plan.json` 和逐阶段原始响应。临时 KGS 在后置 scheduler 为 1/1 healthy/available、active/waiting/checking/broken 为 0 后正常关闭；原 KGS 的预约任务自然结束，调度器恢复 8/8 healthy/available、active/waiting/checking/broken 为 0。该补测只验证 A100 的新结果 Schema 与新 KGS 锁定组合，不替代未完成国产芯片的复测。

## 锁定 KGS 的国产芯片补测

同日以 KG `codex/multiple-device-test@7f2c7cf4`、KGS `main@41101fc`、Gems `4772d816` 及相同 Bundle/候选在昇腾 10.0.0.9、海光、摩尔、沐曦、燧原补测最终 WorkflowResult Schema。各目标的 KGS 均从 Gitee exact commit 建立独立干净 checkout，复用目标原有解释器，不升级 Torch、Triton 或厂商运行时；服务只监听目标 loopback，经 SSH stdio proxy 访问。原 KGS 预约一个 slot，临时 KGS 只可见该卡。下表由本地 `runs/mdt-relu6-latest-JGyCXI/summarize.py` 从三个 `workflow_result.json` 和逐目标原始响应派生，`results.json` 保留完整路径及错误信息。

| 芯片 | 结果 | reference | Preflight | Eval | 正确/总数 | geo mean |
| --- | --- | --- | --- | --- | --- | --- |
| 昇腾 910B4-1 | PASSED | PASSED 15/15 | PASSED | PASSED | 33/33 | 1.0570× |
| 海光 BW | PASSED | PASSED 15/15 | PASSED | PASSED | 33/33 | 1.1339× |
| 摩尔 MTT S5000 | PASSED | PASSED 15/15 | PASSED | PASSED | 33/33 | 0.9886× |
| 沐曦 C550 | PASSED | PASSED 15/15 | PASSED | PASSED | 33/33 | 1.0604× |
| 燧原 ZIXIAOC200 | PREFLIGHT_FAILED | PASSED 15/15 | RUNTIME_ERROR | 未运行 | — | — |

燧原再次在候选首个 timing case 报 `grid.x` 需要 524,288、设备上限 65,535；原 benchmark reference 的 15 个 case 均通过，不能写成 API/dtype 不支持。海光、摩尔、沐曦、燧原的预约 Debug Job 均 `SUCCEEDED`、exit 0，临时 KGS 正常关闭，原 scheduler 分别恢复 7/7、8/8、8/8、1/1 healthy/available，active/waiting/checking/broken/incidents 均为 0。昇腾临时 KGS 也正常关闭，原 scheduler 恢复 8/8 healthy/available 且没有 incident，但本次 600 秒预约在最终清理前返回 `TIMEOUT`；因此昇腾 1.0570× 只保留为原始计时，不作为独占性能结论。全部跨芯片短时计时都不用于不同厂商间的排名。天数和平头哥仍受上述环境阻塞；昆仑芯按用户要求等待新镜像。

测试结束后本次启动的海光、摩尔、沐曦、燧原、昆仑芯和昇腾临时 KGS 均已停止；旧 KGS 的预约 Debug Job 均终态 `SUCCEEDED`、exit 0。旧调度器恢复为：海光 7/7、摩尔 8/8、沐曦 8/8、燧原 1/1、昆仑芯 2/2 healthy/available，active/waiting/checking/broken 均为 0。A100 冒烟同样释放 slot。原有服务、其他用户进程和持久输入证据保留。

配套 host 回归为新 Workflow 及 Gems Definition/审核相关 **41 passed**，包含“reference 总状态为 PASSED 但有 core case 被 skip 时不得进入候选评测”、成功/失败/取消均返回结构化 Workflow Result 的测试。该数字不包含真实设备运行，也不能把天数、平头哥、昆仑芯的未执行阶段算作测试通过。
