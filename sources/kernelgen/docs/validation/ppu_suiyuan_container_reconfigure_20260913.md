# 平头哥与燧原容器配置复验

执行组合：KG v6.3.0@cb511e36 / KGS v6.3.3@265c09e / Protocol v6.2。平头哥配置修正并通过真实 CLI 单卡启动验收；燧原旧容器已恢复，尚不能通过当前 KG CLI 启动 KGS。没有修改软件 backend 实现、升级受保护运行时或移动发布 tag。

## 平头哥

清单登录当前进入 dsw-879514-577d6c8597-zbjft，已经是有 PPU 设备和 SDK 的执行容器；旧内层 codex_fib_thead_20260818 不存在。将 inventory 的 container 改为 `-`，直接在 SSH 登录环境执行，部署目录改为 `/root/xuyao/kernelgen_e2e_20260913`，不创建嵌套容器、不改动其他容器。

默认 python3 提供 Torch 2.9.0、Triton 3.5.0、CUDA 13.0，能识别 16 张 PPU-ZW810E；SDK 工具位于 `/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi`。选择无运行进程、利用率 0 的物理卡 15，使用 thead/triton、2 个请求 worker。临时远端 loopback 21907、本地 loopback 22907，不替换 inventory 的常规端口 18307。

通过项目专用 Gitee 仓库部署公钥准备 exact checkout；私钥只在目标生成，未输出或复制。`/mnt/data` 等共享 FUSE 挂载不支持 chmod，文件保持 777，不适合存储私钥。首次在该目录生成的测试密钥对应公钥已撤销（6053044），私钥已删除且不复用。有效密钥位于清单约定的 `/root/xuyao/kernelgen_server_deploy_ed25519/id_ed25519`（600），仓库部署公钥 ID 6053047；deployment.env.sh 同样为 600。不要把密钥移回 FUSE 共享目录。路径属于当前外层实例；替换外层实例后应重新核验并配置访问，不假设它跟随共享目录自动迁移。

真实 `kg server start`、`status --json`、重复 start、doctor、stop 和停止后 status 全部通过；device_slots=healthy=available=1，active=waiting=checking=broken=0，重复启动 ALREADY_RUNNING，最终 STOPPED。本次临时 KGS 和本地代理已关闭；保留新 checkout 和配置供后续使用，没有遗留模型/Agent 任务。

## 燧原

清单容器 codex_fib_suiyuan_20260820 于 2026-09-08 正常退出（ExitCode=0、OOMKilled=false）。现已重新启动并保留，原镜像、host network、gcu0～7/gcuctl 设备及厂商库挂载不变，无需删除重建。镜像为 harbor.baai.ac.cn/flaggems/suiyuan-flaggems-test-s60-flagtree0.6.1-enflame3.6-pujiang:202608201756。默认 Python 3.12.3，Torch 2.10.0+cpu、torch_gcu 2.10.0+3.7.20260408、FlagTree 0.6.1+enflame3.6。efsmi 能查询 8 张设备，驱动 1.9.10。

补齐清单 deploy_base 下的 deployment.env.sh，引用目标原有的项目只读 key，不打印或复制私钥。容器名称和部署根不变，因此 inventory 的燧原行无需修改。

未通过的边界不再是容器：KG `cli/server.py` 的 DEVICE_ENVIRONMENTS / argparse backend 选项没有 enflame，KGS `management.py` 也未配置燧原设备可见性映射。虽然 KGS 有 backends/enflame 实现，不能用 cuda 别名或绕过单卡隔离假装 CLI 已支持。需要单独补齐管理入口和单卡映射测试，再执行 kg server start 验收；本次只配置容器，未擅自扩展该软件功能。

## 证据与清单验证

本地证据位于 `.kernelgen/experiments/server-start-v630-tJGEhD/`：ppu-reconfigured-test.json 保存全部 CLI 命令与结果，ppu-reconfigured-packages.txt 保存安装包快照与权限检查。旧九机测试 SUMMARY.md 是此前结果，不用本次补测覆盖原始失败记录。

清单变更已通过 `multi_device_validate_record`（支持 jump + container=-）和 shell 语法检查。此处不记录凭据内容，也不把操作记录当作当前资产存活保证；后续仍以 inventory 连接并核验现场。
