# Windows 控制端与 910B 评测

## 角色与边界

Windows 工作区保存控制端仓库并发起远程命令；Ascend 910B 服务器提供设备、CANN、Triton-Ascend 与 KGS 运行环境。Windows 本机没有 `torch_npu` 不代表设备评测不可用，源码和测试合同仍须同步到远端后在设备环境中执行。

设备验证前先用 SSH 配置中的主机别名检查远端：

    ssh <910B主机别名> hostname
    ssh <910B主机别名> git -C /data/hanle/ascend-optimization/Ascend-Optimization status --short --branch

SSH 私钥可由 `ssh-agent` 提供，也可由 SSH 配置中的 `IdentityFile` 直接提供。`ssh-add -l` 只检查当前会话能否访问 agent；它报错不能单独作为 SSH 连接失败的依据。用远端只读命令验证连接。

远端路径、主机别名和容器配置可能变化。每次运行前核对目标仓库、源码 SHA、容器和 KGS 状态；不要只凭两个仓库的分支名相同判断源码一致。不要把私钥、口令、令牌或个人 SSH 配置提交到仓库。

## 910B 运行环境

当前验证环境快照（2026-11-16）：远端仓库位于 `/data/hanle/ascend-optimization/Ascend-Optimization`；设备测试容器为 `tle_yy`，物理卡 7 在容器中通过 `ASCEND_RT_VISIBLE_DEVICES=7` 暴露为逻辑 `npu:0`；设备侧为 Ascend 910B4、CANN 9.0.0、Triton 3.5.1、FlagTree 0.6.0。此快照供诊断环境漂移，实际版本以待测 KGS `/status` 和设备解释器为准。

CANN 初始化、`torch_npu`、测试解释器与设备可见性都属于容器环境。路径 `/usr/local/Ascend/cann-9.0.0` 等必须在目标执行容器内核实；不要因为宿主机路径不同，就判定设备环境不可用。KGS 服务与 `docker exec` 补充检查应使用同一容器和设备映射。

## KGS 复测

1. 检查设备分配、`npu-smi info` 和 KGS `/status`。长驻进程即使当前利用率为零，也可能占用设备；确认不会抢占别的运行后再测。测试按单 worker 串行执行。
2. 若本项目 KGS 服务未运行，只启动有明确项目归属的临时服务。使用独立端口、运行目录、profile 目录和 Triton cache，并设置 `--max-workers 1`。先确认端口空闲，再通过 `/status` 核验协议版本、设备、软件版本和 scheduler 状态。
3. 使用交付目录和显式环境 JSON 准备全新的运行目录：

       python -m hgkda.runtime prepare \
         --operator-directory <远端仓库>/operators/<算子名> \
         --package <已验证的HG-KDA发布目录> \
         --environment <本次运行>/environment.json \
         --run-directory <本次运行>/<算子名>

4. 审查当前源码 SHA 对应的 `source-reviews/<SHA>.json` 后，提交一次完整评测：

       python -m hgkda.runtime evaluate \
         --config <本次运行>/<算子名>/config.json \
         --port <本次KGS端口>

   `evaluate` 会先执行绑定合同与源码 SHA 的 PyTorch/CANN 补充检查，再上传固定合同并请求 KGS 完整评测。以 `audit-final.json`、`evaluation/result.json` 和每项检查结果为准；进程退出码或单个检查通过都不等于完整评测通过。
5. 完成后核验评测 worker 空闲，只终止本次明确启动且 PID/启动身份匹配的 KGS 进程。不要使用 `pkill`、按名称批量结束进程，也不要重启或终止 VCD 服务及共享容器。

复测日志与证据保存在仓库外的独立运行目录，至少保留算子名、源码 SHA、KGS 请求与原始结果、补充检查结果、环境状态和服务生命周期记录。不要把复测结果写进既有发布 ZIP 或覆盖历史成绩；新旧成绩并列报告。

## 最近一次验证

2026-11-16 已按上述边界抽样复测 `scatter_add`、`native_group_norm` 和 `cross_out`。三项 KGS 正式合同及 PyTorch/CANN 补充检查均通过；记录位于服务器 `/data/hanle/ascend-optimization/runs/codex-sample-retest-20261116/`，Windows 侧副本位于工作区外层 `deliveries/hgkda-sample-retest-20261116/`。26 算子代码 ZIP 未修改。
