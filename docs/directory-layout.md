# 工程目录与运行环境

## Windows 控制端

实际 Git 仓库是 E:/0_XSAI/00_KernelAgent/kernelgen/Ascend-Optimization/。外层 kernelgen/ 只是工作目录，不是 Git 仓库，目前仅保留：

    kernelgen/
    ├── Ascend-Optimization/    维护中的 Git 仓库
    └── .control-state/        本地环境、同步回执及恢复资料，不提交

原有外层传输压缩包和提交辅助清单已删除。旧仓库内控制脚本保留在外层 .control-state/recovery/，不再作为运行入口。

## 仓库内

    Ascend-Optimization/
    ├── skills/hgkda/          skill 说明与可调用入口
    ├── src/hgkda/             独立分析核心、分类知识及后端适配
    ├── schemas/              面向模型和网页的 JSON 合同
    ├── examples/             可离线运行的真实证据及 KGS 完整示例
    ├── benchmarks/           三十项算子快照与本轮验证记录
    ├── tests/                自动测试
    ├── tools/                维护及参考构建工具
    ├── docs/                 方法、配置、验证与恢复文档
    ├── manifests/            固定依赖、来源和迁移清单
    ├── sources/              可选 KG/KGS 固定源码及原始分发包
    └── third_party/FlagGems/  固定 FlagGems 快照

Python 核心安装只使用 src/hgkda/，不会顺带安装 KG/KGS、FlagGems、CANN 或模型客户端。sources/ 和 third_party/ 保留用于开发与历史复现，版本以 manifests/ 的锁定清单为准。

## 910B

    /data/hanle/ascend-optimization/
    ├── Ascend-Optimization/   与 GitHub、Windows 同步的维护仓库
    ├── runtime/
    │   ├── hgkda-release-venv/  独立 HG-KDA Python 环境
    │   ├── hgkda-distributions/ 已验证安装包
    │   └── kg-controller/       保留的 KG/模型控制环境
    ├── runs/                 已有实验和本轮验证产物
    └── archive/
        └── Ascend-Optimization-pre-release-20261103/
                              整理前工作树、未提交原件与 Git/LFS

仓库内不再保留旧 project/Ascend-Optimization-v1 活动入口。历史目录可从归档或固定标签恢复；不得直接运行归档内的旧优化控制器。本轮临时 KGS 验证服务已关闭，旧算子优化保持暂停。

## 调用

Windows 使用已安装 hgkda 的 Python；910B 可直接运行：

    /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -m hgkda doctor

模型宿主读取 skills/hgkda/SKILL.md，并调用 scripts/run_hgkda.py 或同一套 hgkda CLI。知识随 Python 包安装；skill 不依赖个人绝对路径或旧 project 目录。
