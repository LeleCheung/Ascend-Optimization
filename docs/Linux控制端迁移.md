# Linux 控制端迁移

## 迁移范围

控制端负责编辑代码、SSH 调度、下载与分析证据、Git 提交同步。KG、Claude Code、KGS、CANN、torch_npu 和 NPU 实验继续运行在 910B；迁移控制端不需要重建容器或复制 controller venv。

仓库已包含导师要求、模型说明、服务器清单、正式控制工具和复现步骤，不依赖原 Windows 外层的 `.goal-transfer/`。当前成果提交见 Git 历史；原始附件和较大压缩包由 LFS 管理。

## 第 1 步：准备工具与仓库

Linux 控制端需要 Python 3.11 或更新版本、Git、Git LFS、OpenSSH 客户端，以及已可用的编辑与终端工具。无需安装 NPU 运行时。

先为新机器配置 GitHub 私有仓库访问权限，然后执行：

```bash
git lfs install
git clone git@github.com:LeleCheung/Ascend-Optimization.git
cd Ascend-Optimization
git lfs pull
git lfs fsck
```

不要复制 Windows Python venv 或整个 `.ssh/`。Linux 使用自己的 SSH 密钥；如需新建，可执行 `ssh-keygen -t ed25519`。GitHub 和 JumpServer 分别配置对应公钥权限。

## 第 2 步：连接 910B

在新机器的 `~/.ssh/config` 中添加：

```sshconfig
Host baai-910b
  HostName bastion.aiops.baai.ac.cn
  Port 2224
  User zhanghanle25z@root@10.0.0.8
```

如使用非默认密钥，再添加 `IdentityFile` 指向自己的私钥。首次连接检查堡垒机主机指纹，然后验证：

```bash
ssh baai-910b hostname
```

成功后确认连接的资产为 910B。SSH 配置、私钥、API key 和本机工具认证保留在机器配置中。控制端迁移不修改 910B Claude 模型配置；接入背景见 [模型说明](模型接入说明.md)。

如果出现 `no available gateway`，表示堡垒机未把连接送到目标资产；有时 SSH 退出码仍为零。先重试连接，持续失败时检查堡垒机网关或联系管理员，不据此判断 910B 程序故障。控制工具会把该响应识别为失败。

## 第 3 步：核验已有成果

```bash
project=project/Ascend-Optimization-v1
python3 "$project/tools/evaluation/verify-master-evidence.py" "$project"
```

本轮已归档证据应显示 274 项核验通过。若之后新增批次证据，该数量可能变化；任一 SHA 不匹配必须先定位原因。

继续阅读 [精简成果表](../project/Ascend-Optimization-v1/operators/本阶段精简汇报-20261010.md)和 [复现与交接](../project/Ascend-Optimization-v1/tools/evaluation/复现与交接-20261010.md)。原始 JSON 中的 Windows/910B 绝对路径是历史采集记录，不批量改写；总审计能将记录中的结果路径解析到本地仓库。

当前最终结果：amin 0.916×，matmul_bias_activation v9 0.884×（两次复验 0.886×、0.884×），narrow_copy 1.102×，完整范围分别为 27、42、33 项，三个算子均超过 0.8×。矩阵乘最新证据见 `project/Ascend-Optimization-v1/operators/matmul_bias_activation/reports/master-closure-20261012/`；数值 Roofline 尚未建立。

## 第 4 步：接续控制与实验

```bash
ctl="$project/tools/control/control.py"
root=/data/hanle/ascend-optimization
python3 "$ctl" --host baai-910b --remote-root "$root" status \
 --batch 20261010 --ports 19655 19657
```

检查 KGS 队列、真实任务状态和设备占用，再安排新批次。保持同卡串行，使用新的 workspace 和报告目录。当前物理卡 7 在服务中为逻辑 `npu:0`；19655 用于 amin/矩阵乘，19657 用于 narrow_copy 设备任务计时。

控制工具示例见 [tools/control](../project/Ascend-Optimization-v1/tools/control/README.md)。KG 启动脚本仍在 910B 上运行，读取 CANN 官方 `set_env.sh` 并使用 controller venv 和服务器 Python，不能直接在控制端执行。唯一仓库、运行目录与评测副本说明见 [910B 目录与复现](910B目录与复现.md)。

需要整夜运行时，在 910B 使用 tmux 托管运行脚本；控制端可断开。调用 `control.py exec` 时，脚本中的命令会在远端执行，工具本身不自动启动或恢复 KG。

## 第 5 步：提交与同步

下载证据、离线核验、审阅变更后，在控制端提交并推送。远端工作树干净时，用以下命令同步已提交源码：

```bash
ssh baai-910b 'git -C /data/hanle/ascend-optimization/Ascend-Optimization fetch origin'
ssh baai-910b 'git -C /data/hanle/ascend-optimization/Ascend-Optimization merge --ff-only origin/main'
ssh baai-910b 'git -C /data/hanle/ascend-optimization/Ascend-Optimization lfs pull'
```

远端存在未提交修改时先审阅，不能直接覆盖。确认控制端 HEAD、GitHub main、远端 HEAD 一致，再检查 LFS 实体。使用 Git 不会自动同步仓库外的 tmux 状态、容器、运行目录和配置。

## 新会话交接

Git 保存工程与资料，不保存聊天或 goal 的应用状态。新会话可从以下任务说明接续：

> 先读 README.md、docs/项目要求与导师沟通.md、docs/ascend-profiling-workflow.md、docs/Linux控制端迁移.md 和本轮复现与交接。当前正式对象是华为列加速比低于 0.8× 的 amin、matmul_bias_activation、narrow_copy，已完成固定 master、KG 无 profiler、KG 原生 profiler和分析优化版四版本闭环。下一阶段优先优化矩阵乘半精度大矩阵。先离线核验证据，再通过 baai-910b 查看真实队列和设备占用；保留完整正确性与逐 case 数据。不要重复启动旧实验或修改历史原始附件。
