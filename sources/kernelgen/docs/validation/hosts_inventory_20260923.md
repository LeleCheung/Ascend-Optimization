# Hosts 清单与 H20 官方镜像验收

## 结论与范围

连接与镜像说明统一到 [tests/hosts.md](../../tests/hosts.md)，原 conf 和 archive 镜像表不再作为独立来源。机器记录位于单一 `kg-hosts` 区块；Fleet、单设备 proxy、批量 proxy 和结果回收脚本只读取该区块。四台昇腾名称分别为 `ascend`、`ascend-8`、`ascend-31`、`ascend-32`，消除重复名称和不完整记录；默认 proxy 端口互不冲突。显式提供的纯文本 inventory 仍可用于原有调用。

KG 分支 `codex/hosts-inventory` 基于 `dev@18a300d8`。配套为 KG v6.5.1 开发分支 / KGS v6.3.4@`376ae33600bc42b5a5f831272b688e438c280123` / Protocol v6.2；没有发布或移动 tag。KGS 通过 Gitee clone 并核对 exact clean checkout，没有复制源码覆盖目标机。未跑模型或 Workflow E2E。

## H20 真机结果

用户修正后的目标为 `115.191.21.142`，SSH 经既定 JumpServer；实际 hostname `baai-sailing-h20-0`，设备 NVIDIA H20-3e，驱动 595.45.04。最终使用用户确认的现成官方 `nvcr.io/nvidia/pytorch:25.04-py3`，digest、镜像 ID、解释器及运行时版本统一记录在 hosts.md，不在多份操作手册重复维护。

新建独立容器 `kernelgen-nvidia-ngc-2504`，`SYS_ADMIN` capability 与 `seccomp=unconfined` 用于 NCU，不修改全局驱动权限。实际 KGS 只使用物理卡 7，逻辑设备 `cuda:0`；1 个 slot、4 个请求线程。该卡与原驻留进程共用已获得用户明确授权，没有停止、重启或修改其他用户进程，因此本轮仅作功能验收，不比较性能或加速比。

KGS 监听目标 loopback 18308，本地通过本分支 SSH stdio proxy 的 loopback 19808 访问；proxy 实际读取新的 hosts.md，非硬编码旧地址。原生输入为安装式 `kernelgenbench/kernelgenbench_square`，候选为纯 Triton 平方 kernel，无近似或 reference 改写。

| 检查 | 结果 |
| --- | --- |
| `/status` | backend=cuda，设备 H20-3e，profile.supported=true |
| `/inspect` | 成功，使用实际返回的 case ID 与 fingerprint |
| `/preflight` | PASSED，9 cases |
| `/evaluate` | PASSED，36/36 |
| `/profile` metrics | completed；NCU report 184728 bytes，normalized metrics 含 square_kernel |
| `/profile` instruction | completed；NCU report 905122 bytes，包含 instruction summary/listing |
| 结束时 scheduler | active=waiting=checking=broken=incidents=0；available=healthy=device_slots=1 |

两次 Profile 均使用 warmup=1、iterations=2，完整产物已下载、核对大小并记录 SHA256。不是只检查 `ncu --version` 或 HTTP 200。临时 KGS 和本地 proxy 已停止，容器及证据保留，未留下 GPU 测试任务。

项目 venv 继承镜像运行时，安装前先检查 pip plan 没有 Torch/Triton/CUDA/NVIDIA 包变更，安装后再次核对全部受保护包版本一致。仅新增轻量依赖 fastapi、uvicorn、annotated-doc、starlette。H20 的 KGS 专用仓库部署公钥已配置，私钥只留在 H20 的项目目录，未复制个人私钥；Git SSH 设置限于该 checkout。

## 华为四台

四台均完成 SSH、Docker、新容器与镜像身份核对，统一使用用户指定的 `huawei-flaggems-test-910b-flagtree0.6.0-ascend3.5:202608191554`。新容器均为 `kernelgen-ascend-flagtree060`，镜像 ID 相同，实际 Python/包元数据一致。10.0.0.9 另检查 Torch、torch_npu、Triton 可导入。

本轮没有在四台重新进行 GPU/NPU Eval/Profile，不把镜像和连接检查写成全平台执行通过。旧 `tle_yy`、旧 KGS 容器及任务保持不动，新记录的端口只是后续显式部署默认值。

## 回归、证据与已停止尝试

host 回归 72 passed，覆盖 Markdown 区块识别、缺失/重复/未闭合区块拒绝、记录唯一性、端口唯一性、Fleet 解析、纯文本 inventory 以及 SSH proxy。两个 proxy 对重复设备名拒绝选择，避免取第一条而连错机器。相关 Shell 脚本通过 `bash -n`，测试前核对 KG 导入来自本 worktree。

本地原始证据位于 KG 主工作区 `runs/hosts-inventory-20260923/`，包括各机器容器身份、H20 KGS 启动与版本快照、请求结果、Profile 产物哈希和收尾记录。目标端证据位于项目目录 `h20-ncu-validation/`。主工作区原有未提交文件及暂存区未被清理或覆盖，PR 在独立 worktree 开发。

最初 inventory 误填的 A100 地址仅用于连接/镜像检查，其临时 24.09 容器已停止，未将那次结果用于 H20 验收。26.03 镜像拉取过程中，指定代理对 NVCR 返回 500，直连下载较慢；用户随后确认使用现成 25.04。26.03 下载与任务专用 loopback registry 已停止，部分下载文件保留；未修改系统全局代理或 Docker daemon 配置，未把代理凭据写入文档或 Git。
