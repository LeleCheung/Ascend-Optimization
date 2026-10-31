# KG与KGS离线备份说明

2026-10-27：现有源码可以离线保留和继续使用，不依赖再次访问上游。当前项目保存KG 6.7.0、KGS 6.5.0及原始发布包；没有核实为当前上游最新dev／master。

## 已有原件

- sources/kernelgen/：本项目使用的KG源码快照。
- sources/kernelgen_server/：本项目使用的KGS源码快照。
- sources/archives/kernelgen-dev.zip：原始KG压缩包，26,666,826字节。
- sources/archives/kernelgen_server-v6.5.0.tar：原始KGS归档，82,135,040字节。
- manifests/components.lock.yaml、manifests/hgkda-stage2.lock.json：源码与运行环境的锁定信息。

备份前已检查Windows与910B的sources目录，未发现Git LFS指针替代实体内容。两端分别保存快照，Linux运行缓存和符号链接没有覆盖Windows原件。

## 独立备份位置

Windows仓库外：

- E:\0_XSAI\00_KernelAgent\backups\kg-kgs-20261027T065108Z-windows：Windows快照，共12,978个文件，归档约52.4 MB。
- E:\0_XSAI\00_KernelAgent\backups\kg-kgs-20261027T065229Z-910b：910B快照的下载副本，共13,225个常规文件、11个符号链接，归档约53.3 MB。

910B仓库外：

- /data/hanle/ascend-optimization/archive/code-backups/kg-kgs-20261027T065229Z-910b

每个备份含kg-kgs-snapshot.tar.gz、manifest.json和中文说明。清单记录逐文件SHA256、链接目标和整包SHA256；创建归档后逐成员重读核验。910B下载副本还独立校验传输字节与全部成员。

| 快照 | 压缩包SHA256 |
|---|---|
| Windows | 0d335c4c1f2bdb21ba7f84d6071dc29db1b715986564c4744d87c5f77898e6a6 |
| 910B | 10f080f06ddca1dd730697cd018edd99f82d9634f18a40e36478997210dd2aa4 |

## 恢复

1. 使用SHA256工具核对整个压缩包与manifest.json中的archive_sha256。
2. 解压至新的空目录，不覆盖当前运行环境。Linux使用tar -xzf kg-kgs-snapshot.tar.gz -C 新目录，保留符号链接。
3. 按manifest.json逐文件核对SHA256、大小及链接目标。
4. 按附带组件清单准备固定环境，再做独立KG/KGS烟雾测试。

这是源码及原始包备份，不包含容器镜像、完整Python环境、模型凭据、上游Git历史或全部算子实验结果。已有容器与完整实验仓库继续保留。后续获得上游权限时，另建新分支验证环境，不覆盖旧实验的固定版本。

## 上游访问检查

本次Windows访问Gitee报Host key verification failed，910B报Permission denied (publickey)。前者属于SSH主机身份校验，后者表示公钥认证未通过；这些结果不足以判断仓库权限是否变化，更不能据此判断合作状态。

该备份没有新增Git提交或推送，也没有修改运行服务或共享镜像。
