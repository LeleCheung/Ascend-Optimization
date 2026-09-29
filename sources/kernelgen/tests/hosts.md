# 多芯片机器、镜像与连接清单

本文件是机器连接和镜像配置的唯一维护入口，合并原 `multi_device_batch_hosts.conf` 与 `docs/archive/deployment/vendor_images.md`。脚本只读取下面的 `kg-hosts` 代码块，不解析正文中的镜像表或命令。更新时间：2026-09-23。

连接配置不等于 KGS 已启动，也不分配设备。启动前查询设备占用和已有 KGS；不能仅凭利用率为 0 就接管其他服务的卡。已有 CLI 实例继续使用其保存的配置，更新本文件不会改变旧实例或旧 campaign。部署步骤见 [远端 KGS 指南](../docs/operations/deployment/remote_server.md)，实验步骤见 [多设备手册](../docs/operations/multiple_device_experiment_runbook.md)。

## 1. 机器可读连接记录

九列为 `name|mode|address|ssh_port|container|deploy_base|server_port|devices|jump_login`。`devices` 留空表示本文件未指定授权卡号，不表示允许使用全部卡；`container=-` 表示直接使用 SSH 登录环境。名称必须唯一。`ascend` 保留为 10.0.0.9 的名称，其余三台使用 `ascend-8`、`ascend-31`、`ascend-32`。

```kg-hosts
# name|mode|address|ssh_port|container|deploy_base|server_port|devices|jump_login
nvidia|jump|115.191.21.142|2224|kernelgen-nvidia-ngc-2504|/data/xuyao/kernelgen_hosts_20260923|18308||xuyao@secure@115.191.21.142
tianshu|jump|10.31.28.29|2224|codex_fib_iluvatar_20260728|/workspace/kernelgen_e2e_20260728|18301||xuyao@secure@10.31.28.29
hygon|jump|10.232.2.26|2224|codex_fib_hygon_20260728|/workspace/kernelgen_e2e_20260728|22102||xuyao@secure@10.232.2.26
musa|jump|10.121.38.9|2224|codex_fib_mthreads_20260728|/workspace/kernelgen_e2e_20260728|22103||xuyao@secure@10.121.38.9
metax|jump|192.168.2.115|2224|codex_fib_metax_20260829|/data/xuyao/kernelgen_e2e_20260829|22104||xuyao@secure@192.168.2.115
kunlun|jump|10.21.1.77|2224|codex_fib_kunlunxin_20260901|/home/secure/xuyao/kernelgen_e2e_kunlun_20260901|18305||xuyao@secure@10.21.1.77
ascend|jump|10.0.0.9|2224|kernelgen-ascend-flagtree060|/home/secure/xuyao/kernelgen_hosts_20260923|18306||xuyao@secure@10.0.0.9
ascend-8|jump|10.0.0.8|2224|kernelgen-ascend-flagtree060|/home/secure/xuyao/kernelgen_hosts_20260923|18316||xuyao@secure@10.0.0.8
ascend-31|jump|10.0.0.31|2224|kernelgen-ascend-flagtree060|/home/secure/xuyao/kernelgen_hosts_20260923|18326||xuyao@secure@10.0.0.31
ascend-32|jump|10.0.0.32|2224|kernelgen-ascend-flagtree060|/home/secure/xuyao/kernelgen_hosts_20260923|18336||xuyao@secure@10.0.0.32
ppu|jump|8.130.132.221|2224|-|/root/xuyao/kernelgen_e2e_20260913|22107||xuyao#root#3be1de7b-d9e4-4e2b-8022-a4084e2f1447
suiyuan|jump|10.12.1.38|2224|codex_fib_suiyuan_20260820|/home/secure/xuyao/kernelgen_e2e_suiyuan_20260901|18309||xuyao@secure@10.12.1.38
```

SSH 堡垒机为 `bastion.aiops.baai.ac.cn`，例如：

```bash
ssh -i /root/.ssh/id_ed25519 -p 2224 'xuyao@secure@115.191.21.142@bastion.aiops.baai.ac.cn'
```

这是当前账号示例，不是公共凭据。其他账号可通过 proxy 的 `--jump-user` 或 `KG_JUMP_USER` 替换，身份文件可用 `--identity` 指定。SSH 端口 2224 与 KGS HTTP 端口不同；KGS 只监听目标环境 loopback，通过 SSH stdio HTTP proxy 访问，不开放远端公网端口。

`scripts/proxy/start_all.sh` 默认本地端口为 `18000 + server_port 的末两位`；上述十二条记录的本地端口互不冲突。命名实例的 `--listen-port` 仍可独立指定。华为新容器的端口是新部署默认值，不迁移或停止旧 22106 等服务。

## 2. 本轮核验的 NVIDIA 和华为镜像

### NVIDIA：115.191.21.142 / H20

- 硬件：8 × NVIDIA H20-3e；现场驱动 595.45.04。
- 官方镜像：`nvcr.io/nvidia/pytorch:25.04-py3`，使用 H20 已有缓存，不使用浮动 latest。
- Registry digest：`nvcr.io/nvidia/pytorch@sha256:d1eac6220dd98ef5870b1a76673cfb6f84451135a6d8a174cb92258a6bf4576d`；镜像 ID：`sha256:43e6802c38a508fb7fb61b2733b7617550d221d03f97e0f6ebc8fc858dafc467`。
- 实查版本：Python 3.12.3、Torch `2.7.0a0+79aa17489c.nv25.04`、CUDA 12.9、Triton 3.2.0、NCU 2025.2.0.0。
- 不沿用此前误填的 A100 地址 `10.1.2.106`，也不把 A100 上的自制 `kernelgen-nvidia-cu132-nsight2026.1:base` 当作官方镜像。

[NVIDIA PyTorch 25.04 发布说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-04.html)是镜像来源依据；实际包版本以上述镜像内检查为准。用户确认使用该现成官方镜像，不再等待 26.03 下载。实际 profiling 可用性仍须由 H20 真机验证，不能仅依据镜像标签或 `ncu --version` 判断。

为本项目使用独立的 `kernelgen-nvidia-ngc-2504` 容器，不替换其他用户容器。NCU 需要设备性能计数器权限；仅给新容器 `SYS_ADMIN` capability 和 `seccomp=unconfined`，不修改宿主驱动的全局权限设置。代理认证信息不保存在本文件中。

```bash
IMAGE=nvcr.io/nvidia/pytorch:25.04-py3
CONTAINER=kernelgen-nvidia-ngc-2504
BASE=/data/xuyao/kernelgen_hosts_20260923
docker run -dit --name "$CONTAINER" --network=host --ipc=host \
  --gpus all --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
  --shm-size=16g --ulimit memlock=-1 \
  -v "$BASE:$BASE" -v /etc/localtime:/etc/localtime:ro \
  "$IMAGE" bash
```

容器可见设备不等于任务授权。此次经用户明确授权，临时共用 7 号卡，不停止占用全部八卡的驻留 Python 进程；该授权不自动延续到后续实验。KGS 单 slot 验证 `kernelgenbench_square`：Preflight 9/9、Eval 36/36、NCU metrics/instruction 均 completed，产物中确认候选 kernel。共用卡上的计时不作为性能结论。临时 KGS/proxy 已停止，容器保留供后续显式启动。详见 [验收记录](../docs/validation/hosts_inventory_20260923.md)。

KGS 项目解释器为 `/data/xuyao/kernelgen_hosts_20260923/venv/bin/python`，使用 `--system-site-packages` 继承镜像运行时，仅补充轻量服务依赖；配套 checkout 位于同一目录下的 `kgs/`。使用 `kg server start` 时显式传入这两个路径，避免误用另一个环境。

### 华为：10.0.0.9、10.0.0.8、10.0.0.31、10.0.0.32

四台统一使用用户指定镜像：

```text
harbor.baai.ac.cn/flaggems/huawei-flaggems-test-910b-flagtree0.6.0-ascend3.5:202608191554
```

已在四台新建同名独立容器 `kernelgen-ascend-flagtree060`。镜像 ID 均为 `sha256:91c4f9f3d90e0ccd6b7dfc263952b1d1a14dd1af85852ffeef865f73cf721b64`，registry digest 为 `sha256:b0c030e19f0e17c619512a9d0a5e5a66a5786a158ef773dd12b4d66f9131f81e`。四台 Python 为 `/usr/local/python3.11.15/bin/python3`，Torch `2.9.0+cpu`、torch_npu `2.9.0.post2`；Torch 的 `+cpu` 标签不表示 NPU 不可用，NPU 扩展由 torch_npu 提供。

本轮确认连接、镜像身份和解释器，不把这些检查宣称为四台均已完成 Eval/Profile。10.0.0.8 的 `tle_yy` 和其他原有容器未修改，存量任务不迁移。旧镜像 tag `202607091608`、无 tag 的镜像别名和不同的 FlagTree 镜像均不能代替上述固定镜像。

以下为创建新容器的模板，先确认名称未占用、宿主目录存在；不要照抄命令覆盖已有容器：

```bash
IMAGE=harbor.baai.ac.cn/flaggems/huawei-flaggems-test-910b-flagtree0.6.0-ascend3.5:202608191554
CONTAINER=kernelgen-ascend-flagtree060
BASE=/home/secure/xuyao/kernelgen_hosts_20260923
docker run -dit --name "$CONTAINER" --network=host --ipc=host --privileged \
  --shm-size=16g --ulimit memlock=-1 \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/localtime:/etc/localtime:ro -v "$BASE:$BASE" \
  "$IMAGE" bash
```

仅挂载现场存在的辅助目录；设备节点必须核对。以上是容器准备，不会安装 KGS、分配 NPU slot 或启动优化。KGS 使用 `backend=npu` 和 `timing=profiler`。

## 3. 其他厂商镜像与启动参考

以下内容迁自旧镜像表，原始记录截至 2026-09-01；本轮未逐台重新拉取或验证。表中历史驱动不是当前主机驱动的权威值，镜像也不一定等于连接记录中存量容器的来源。安装或升级 Torch、Triton/FlagTree、厂商运行时需另行确认，不能把历史建议当作本次操作授权。

| 厂商 | 历史型号 / 驱动 | 镜像或环境来源 |
| --- | --- | --- |
| 天数 | BI-V150 / 4.4.0 | `harbor.baai.ac.cn/flaggems/iluvatar-flaggems-test-bi-v150:20260819` |
| 海光 | BW1000 / 6.3.28-V1.3.0b | `harbor.baai.ac.cn/flaggems/hygon-flaggems-test-bw1000-flagtree3.6-pujiang:202608181520` |
| 摩尔线程 | MTT S5000 / 3.3.3-server | `harbor.baai.ac.cn/flaggems/mthreads-flaggems-test-mtt-s5000-flagtree3.6-pujiang:202608181700` |
| 沐曦 | C550 / 3.3.12 | `harbor.baai.ac.cn/flaggems/metax-flaggems-test-c550-flagtree0.6.1_metax3.6-pujiang:202608191108` |
| 平头哥 | PPU-ZW810E / 1.3.2-d7f5a2 | 运维提供的外层容器，inventory 的 `container=-` 不代表裸宿主机 |
| 寒武纪 | MLU590-M9DE / v6.2.15 | `harbor.baai.ac.cn/flaggems/cambricon-flaggems-test-mlu590-m9de-triton3-2-1:20260618`；尚无可用连接记录 |
| 昆仑芯 | P800 / 515.58 | `harbor.baai.ac.cn/flagos-inner-models-release/kernelgen-xpu:v1` |
| 燧原 | S60 / 1.10.6 | `harbor.baai.ac.cn/flaggems/suiyuan-flaggems-test-s60-flagtree0.6.1-enflame3.6-pujiang:202608201756` |
| 清微智能 | 未记录 | 未提供镜像和连接记录 |

这些模板统一先设置对应的 `IMAGE` 和未占用的 `CONTAINER` 名称；检查挂载路径和设备节点后使用。`docker pull "$IMAGE"` 只拉镜像，`docker exec -it "$CONTAINER" bash` 进入已有容器。不要对 inventory 中已运行的容器重复执行 `docker run`。

### 天数

```bash
docker run -dit --name "$CONTAINER" --network=host --pid=host --privileged \
  --cap-add=ALL --security-opt seccomp=unconfined \
  -v /lib/modules:/lib/modules -v /dev:/dev \
  -v /etc/localtime:/etc/localtime:ro \
  -v /data1:/data1 -v /home:/home -v /tmp:/tmp -w /root "$IMAGE"
```

### 海光

```bash
docker run -dit --name "$CONTAINER" --network=host --ipc=host --privileged \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  -v /opt/hyhal:/opt/hyhal -v /etc/localtime:/etc/localtime:ro \
  -v /data:/data -v /home:/home -v /tmp:/tmp -w /root "$IMAGE" bash
```

### 摩尔线程

```bash
docker run -dit --name "$CONTAINER" --network=host --pid=host --privileged \
  --cap-add=SYS_PTRACE --shm-size=16g --security-opt seccomp=unconfined \
  -e MTHREADS_VISIBLE_DEVICES=all -e MTHREADS_DRIVER_CAPABILITIES=all \
  -v /etc/localtime:/etc/localtime:ro -v /data:/data -v /home:/home \
  -w /root "$IMAGE" bash
```

### 沐曦

```bash
docker run -dit --name "$CONTAINER" --network=host --uts=host --ipc=host --privileged \
  --group-add video --shm-size=100g --ulimit memlock=-1 \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --device=/dev/dri --device=/dev/mxcd \
  -v /etc/localtime:/etc/localtime:ro -v /data:/data -v /home:/home -v /tmp:/tmp \
  -w /root "$IMAGE" bash
```

### 平头哥

当前 SSH 入口已在外层容器中，不默认嵌套 Docker。历史共享 wheel 路径为 `/mnt/data/jhw/flaggems/flagtree-0.5.0+ppu.git2c2cf04c-cp312-cp312-linux_x86_64.whl`，仅作为来源记录，不在本轮安装或移动该文件。已发现根 EXT4 文件系统故障；独立验证使用 `/mnt/workspace` NAS 保存源码、TMPDIR、编译缓存及产物。连接记录保留既有服务目录，不表示该故障已修复。

手工运行最新 Gems pytest 时，要把 `TMPDIR`、`XDG_CACHE_HOME`、`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR` 和 **`FLAGGEMS_CACHE_DIR`** 都指向 NAS；只设置前四项仍会让动态点算子源码写进根分区的 `~/.flaggems/code_cache`。当前容器的 Triton 编译器还需进程级 `CUDA_PATH=/usr/local/cuda`、`PPU_SDK=/usr/local/PPU_SDK`。手工 pytest 可用 `GEMS_VENDOR=thead` 选择厂商，KGS adapter 则按 `backend=thead` 自动注入；不把这些变量写成系统全局配置。最新 Gems 的实际验证和 KGS 认证缺口见 [跨芯片记录](../docs/validation/multiple_device_relu6_20260925.md)。

### 寒武纪

先人工确认需要映射的 `/dev/cambricon*` 设备，不使用旧表中未定义的 `$devs` 直接启动：

```bash
docker run -dit --name "$CONTAINER" --network=host --privileged \
  --cap-add=SYS_PTRACE --shm-size=32g \
  --device=/dev/cambricon_ctl --device=/dev/cambricon0 \
  -v /usr/bin/cnmon:/usr/bin/cnmon:ro -v /etc/hosts:/etc/hosts:ro \
  -v "$PWD:/cambricon" -w /cambricon "$IMAGE" bash
```

### 昆仑芯

```bash
docker run -dit --name "$CONTAINER" --network=host --privileged \
  --ulimit stack=67108864 --ulimit memlock=-1 --ulimit nofile=120000 --shm-size=256g \
  --group-add video --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
  --device=/dev/xpu0 --device=/dev/xpu1 --device=/dev/xpu2 --device=/dev/xpu3 \
  --device=/dev/xpu4 --device=/dev/xpu5 --device=/dev/xpu6 --device=/dev/xpu7 \
  --device=/dev/xpuctrl --device=/dev/fuse \
  -v /etc/localtime:/etc/localtime:ro -v /data:/data -v /home:/home -v /tmp:/tmp \
  -w /root "$IMAGE" bash
```

昆仑芯当前有独立运行时兼容阻塞，不因连接记录存在就启动批量优化。

### 燧原

```bash
docker run -dit --name "$CONTAINER" --privileged \
  -v /etc/localtime:/etc/localtime:ro -v /data:/data -v /home:/home \
  -w /root "$IMAGE" bash
```

这是历史模板；实际 KGS 使用 `backend=enflame`、Torch 设备类型 `gcu`，并通过 `TOPS_VISIBLE_DEVICES` 绑定授权卡。Profiler 报告是否可用以 `/status` 能力和真实 `/profile` 为准。

## 4. 代理、凭据和持久化

本地可复用设置放在 Git 忽略的 `.kernelgen/local-proxies.env`，权限 600，新 clone 需单独配置。禁止打印、提交或在 shell tracing 下加载。`KG_GITHUB_PROXY_NVIDIA`、`KG_GITHUB_PROXY_OTHER`、`KG_GITEE_PROXY_NVIDIA` 按目标配置；旧 H800 使用的 `10.6.212.22:3128` / `:2080` 是历史地址，不能因为名称仍为 nvidia 就套用到新 H20。本轮用户指定的 H20 拉取代理地址为 `http://114.111.19.82:80`，认证只保留在本地安全配置中。

按仓库或单次命令应用代理，不设置系统全局 HTTP(S)_PROXY。`FLAGOS_HARBOR_PASSWORD` 只经 `docker login --password-stdin` 传入，不出现在 argv 或日志。Docker daemon 的出口与 shell Git 代理不是同一个配置，不为一次拉取失败修改全局 daemon 或其他用户容器。

远端部署凭据放在宿主持久目录 `<deploy_base>/deployment.env.sh`，权限 600、不得提交，容器需挂载可见；用 `kg server start/configure --remote-env-file <容器内绝对路径>` 显式指定，KG 不自动加载或上传本地代理文件。部署环境文件不用于给后台 KGS 注入临时运行时配置。

已有专用 Gitee 只读 key 的历史宿主路径如下。`kernelgen_server_deploy_ed25519` 是目录，`known_hosts` 位于私钥旁边；验证实际存在和容器可读后使用，不复制其他账号私钥，不以旧容器内 key 代替持久 key：

- 天数、沐曦：`/home/secure/xuyao/kernelgen_server_deploy_ed25519/id_ed25519`。
- 海光、摩尔：`/data/xuyao/kernelgen_server_deploy_ed25519/id_ed25519`。
- 平头哥外层环境：`/root/xuyao/kernelgen_server_deploy_ed25519/id_ed25519`。
- NVIDIA H20：`/data/xuyao/kernelgen_hosts_20260923/kgs-deploy-key/id_ed25519`，仅授权 KGS 仓库，私钥保留在 H20；容器通过项目目录挂载访问。
- 新华为容器的凭据挂载需按各自宿主现状配置，不假定旧部署目录已经挂入新容器。

Gitee SSH 固定使用端口 22、BatchMode、IdentitiesOnly 和 host-key 校验；不要沿用某些旧容器的 62262 默认端口。KGS 源码经 Gitee 获取，exact commit 由 [KG lock](../deployment/kgs.lock.yaml) 决定；镜像、KGS release 和 Protocol 是不同概念，不用镜像 tag 推断协议能力。
