# 昇腾算子优化复现仓库

本仓库把昇腾算子优化实验所需的项目文件、第三方源码、固定版本信息和复现记录集中保存。目标是在一台新的 910B 环境中，依据本文档重建 KernelGen、KernelGen Server（KGS）和 FlagGems 的运行环境，并复现实验。

## 目录

- `project/KernelGen/`：本实验的脚本、候选算子、报告和说明文档。
- `third_party/FlagGems/`：实验固定版本的 FlagGems 源码。
- `sources/kernelgen/`：KernelGen 6.7.0 解压源码。
- `sources/kernelgen_server/`：KernelGen Server 6.5.0 源码。
- `runtime/catalogs/`：实验使用的 Catalog；运行日志和大型临时文件不在仓库中。
- `manifests/`：组件提交号、镜像和软件版本。

## 复现原则

本仓库不提交 Python 虚拟环境、容器层或 Claude 原生二进制。它们依赖机器架构和驱动，无法作为跨机器的可靠复现材料。请按照 `manifests/components.lock.yaml` 创建环境，并使用 `project/KernelGen/experiments/` 下的脚本运行实验。

FlagGems、KernelGen 和 KGS 的源码目录均为导出副本，不包含上游仓库的 `.git` 元数据；版本和来源记录在清单中。这样克隆本仓库后可以直接查看和构建全部源码，同时不会产生嵌套 Git 仓库。
