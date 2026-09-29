# 远端 KGS 部署

本指南面向“本地 KG + 远端 KGS”。KG 和模型 Runtime 不需要部署到目标 GPU/NPU 容器；KGS 在已有目标执行环境中运行，仅监听 loopback。本指南不创建容器、不分配设备，也不要求开发机先 clone 一份 KGS checkout。

## 配置和事实各从哪里读取

| 信息 | 来源与适用边界 |
| --- | --- |
| 新实例 KGS repository、release、exact commit | 当前 KG 的 [deployment/kgs.lock.yaml](../../../deployment/kgs.lock.yaml) |
| SSH 入口、已有容器、部署目录、远端端口 | [tests/hosts.md](../../../tests/hosts.md) 的 `kg-hosts` 九列记录；不是现场存活证明 |
| 已创建命名实例的参数 | 创建该实例的 CLI home 下 `servers/<name>/config.json`；更改 inventory 不会自动更新实例 |
| 实际设备、backend、timing、scheduler、协议及能力 | `kg server status <name> --json` 中的 `server`，即 KGS `/status` |
| 镜像、驱动和设备是否可用 | 操作方对目标现状的核验；历史镜像表不能替代实测 |
| Gems 默认来源策略 | 锁定 KGS 的 compatibility.yaml；branch policy 在实验准备阶段解析 HEAD，实际 commit 记录在实例配置 |

始终在同一 KG 工作目录执行管理和任务命令，或固定 KERNELGEN_CLI_HOME。从别的目录启动会使用另一份本地状态。不要直接手改 config.json。

## 三种端口不是同一个端口

| 参数 | 监听位置 | 用途 |
| --- | --- | --- |
| SSH 命令中的 -p | SSH/JumpServer 入口 | 建立登录会话，不是 HTTP |
| --remote-port | 目标 KGS 执行环境的 127.0.0.1 | KGS HTTP 服务；inventory 的 server_port |
| --listen-port | 运行 KG 的机器的 127.0.0.1 | 本地 HTTP 代理，kg run 的 --eval-server 使用此地址 |

kg server start 不按设备名猜端口，两种 HTTP 端口必须显式配置。本地端口可自行选择未占用值，不要求等于远端端口。文档中的 19606 是示例，不是昇腾保留端口。

独立 [proxy 批量脚本](../../../scripts/proxy/README.md) 默认按 18000 + 远端端口末两位映射本地端口；这只是该脚本的约定，与命名实例的 --listen-port 无关。Web 的 /api/runs 不属于 KGS API，不应通过这些 KGS 代理地址查询。

## 容器、镜像与解释器

--container 接收已运行的容器名，不是 Docker 镜像名；KG 使用 sudo docker exec -i 进入它，不执行 docker pull/run。--container - 表示直接使用 SSH 登录到的环境，该环境也可能已经处于外层容器。不要仅按厂商名判断是否需要内层容器；以机器清单和实际拓扑为准。

--remote-python 选择该环境里可执行的 Python，建议显式填绝对路径。目标必须预装匹配驱动的 Torch、Triton、厂商扩展和 profiler；KG 只准备配套 KGS 轻量依赖，不修补错误镜像或升级受保护运行时。虚拟环境与镜像选择是部署侧职责，按任务动态切换环境仍是 [后续设计](../../design/workflows/catalog_optimize.md#后续设计一个-kgs-管理多个执行环境)。

若容器默认 Python 因 PEP 668 拒绝安装，使用项目虚拟环境并通过 `--remote-python <venv>/bin/python` 选择它，不使用 `--break-system-packages`。`python3 -m venv --system-site-packages <venv>` 可继承现有厂商运行时；若缺少 ensurepip 但系统已有 pip，可用 `--without-pip --system-site-packages` 创建环境，再确认 `<venv>/bin/python -m pip --version` 可用。燧原的实际验证见 [Enflame CLI 单卡验收](../../validation/enflame_cli_20260913.md)。

`--backend` 是厂商身份，不直接等于 Torch 设备类型：NVIDIA 用 `cuda`，昇腾用 `npu`，燧原用 `enflame`，其他厂商按 `kg server start --help` 与配套 KGS 部署教程选择。燧原要求目标 Python 已安装 `torch_gcu`，通过 `TOPS_VISIBLE_DEVICES` 限定物理卡；例如 `--backend enflame --devices 7 --timing triton` 只允许使用物理卡 7，KGS 中映射为 `gcu:0`。管理层会清除继承的其他设备可见性变量，不用 CUDA 别名模拟燧原。昇腾正式计时用 `profiler`，已验证的其他厂商通常用 `triton`；最终仍检查目标 `/status`，不能看到兼容 CUDA 接口就当成 NVIDIA。可见设备映射和强探针自测遵循配套 KGS 教程。

机器连接、当前 NVIDIA/华为镜像以及其他厂商的历史镜像模板统一维护在 [hosts.md](../../../tests/hosts.md)。各节明确标注实际核验范围，历史模板不保证匹配当前驱动或现有容器，也不授权替换 Triton/FlagTree。新部署核对镜像 digest、挂载和运行时，验证结果另记报告；不在多个文档复制“当前镜像”表。

## 首次启动

先完成 [KG 安装](../../ONBOARDING.md#2-安装) 和模型 Runtime 认证。远端部署不需要运行本地 KGS 服务，也不需要为部署检查准备本地 KGS checkout；KG 调用 Workflow 所需的 `kernelgen_client` 会随 KG 安装，不能把“无本地部署 checkout”理解成移除了客户端依赖。

按 inventory 和已授权空闲卡填入以下模板；这是参数模板，不是某台机器的现状。--ssh-command 只包含 SSH 基础登录命令，不能包含远端命令、TTY、密码或 token：

```bash
kg server start <new-instance> --target remote \
  --backend <backend> --devices <authorized-device-list> --timing <timing> \
  --remote-port <inventory-server-port> --listen-port <free-local-port> \
  --ssh-command 'ssh <approved-login-options-and-destination>' \
  --container <inventory-container> --remote-python <absolute-python-path> \
  --remote-kgs-root <container-visible-new-checkout-path> \
  --remote-state-root <container-visible-instance-state-path> \
  --remote-env-file <container-visible-deployment.env.sh>
kg server status <new-instance>
```

后面三项路径按实际部署配置；省略 remote-kgs-root / remote-state-root 时默认位于目标用户的 ~/.kernelgen/ 下，remote-env-file 可选且不会自动发现。持久化路径必须在容器内可见，不把宿主机路径直接当成容器路径。

KG 从当前 lock 在目标容器内经 Gitee 准备 exact clean checkout，再调用该 checkout 的管理入口安装轻量依赖、启动 KGS 和本地 SSH stdio mux proxy，最后校验 readiness。不要用 scp/tar 覆盖远端源码。checkout 不匹配时保留并报告，不 reset 或接管未知仓库。

readiness 还核对 /status.server_version 与已配置实例的 release 一致，作用是识别是否连到了错误实例，不用该字段推导协议或 Bundle 能力。Protocol 只由 api_version 判断。

目标访问 Gitee 与本机登录目标机是两套认证。项目专用只读部署 key 保存在宿主机持久化目录，路径见 inventory 注释，容器需能访问；kernelgen_server_deploy_ed25519 是目录，私钥为其下 id_ed25519。机器侧 deployment.env.sh 引用已有 key/known_hosts 并按仓库配置代理，权限 600、不提交、不打印。Gitee SSH 固定端口 22，启用 BatchMode、IdentitiesOnly 和主机密钥校验，不复用历史容器的默认 62262 端口。

本地 .kernelgen/local-proxies.env 不自动加载或上传。--remote-env-file 仅在部署操作期间加载，不注入后台 KGS；不能用它给运行时临时补厂商环境。需要的环境应由目标解释器/执行环境预先提供。

## 后续管理与升级

```bash
kg server start <name>
kg server status <name> --json
kg server logs <name>
```

重复 start 按实例已保存的 commit 和连接参数幂等恢复，不随 KG 升级自动更新。已有实例需要改端口、容器或解释器时，先确认没有活动任务，再 stop、configure、start；release 切换使用新实例名。新旧实例不得重叠占用同一物理卡，即使当前 idle 也不代表设备所有权已释放。

需要 Gems 时按 [实验准备](../multiple_device_experiment_runbook.md#2-启动并检查-kgs) 执行 install-flaggems。Server 不绑定 Catalog；任务选择 Catalog 或上传 Bundle。Bundle 可执行性以目标 capability 为准，不由 KGS release 猜测。

## 已有非受管 Server 与连接排障

仅为已有 Server 建立代理，不接管其生命周期时，在 KG 根目录运行：

```bash
bash scripts/remote_server/run_persistent_remote_http_proxy.sh \
  --device <inventory-name> --listen-port <free-local-port>
curl --noproxy '*' -fsS http://127.0.0.1:<free-local-port>/status
```

未传 --remote-port 时脚本读取 inventory；停止该前台代理不会停止远端 Server。不要再为命名实例重复启动一个代理，也不要使用 ssh -L 或暴露公网 KGS 端口。

若有全局 HTTP 代理，curl 显式使用 --noproxy '*'；Python/Agent 进程排除 loopback，示例仅影响该进程：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  kg server status <name>
```

502 先区分全局 Squid 与 SSH/Server 故障；代理存活不等于 KGS 正常。KGS 真正健康还须核对 scheduler。断链不会自动重放在途 Eval/Profile/Debug，防止重复执行；下一条新请求才重建会话。服务端设备自测见配套 KGS 的 docs/multiple_device_deploy_tutorial.md，其中旧环境描述不覆盖本指南的 inventory 与版本选择规则。
