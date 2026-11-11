# 工程目录与运行环境

## Windows 控制端

实际Git仓库是E:/0_XSAI/00_KernelAgent/kernelgen/Ascend-Optimization/。外层kernelgen/只是工作目录，不是Git仓库：

    kernelgen/
    ├── Ascend-Optimization/    维护中的 Git 仓库
    ├── .control-state/        当前Python环境与本机归档，不提交
    │   ├── release-venv/     仍在使用的Windows Python环境，保持原路径
    │   ├── archive/20261110/ 一次性脚本、回执、发布包及实验副本
    │   └── recovery/         更早的恢复原件，保持原路径
    └── hooks/                外层已有Git LFS钩子；不属于HG-KDA源码

外层归档中的脚本仅供追溯，不再作为运行入口。`archive/20261110/manifest.json`记录整理前路径、归档位置与文件SHA；原文件内容不改写。虚拟环境含绝对路径，未直接移动；归档中的历史wheel路径不影响已安装包的导入。历史记录中的旧绝对路径仍保留，定位时按迁移清单查询。

可复用的远程Python执行工具已收敛到`tools/maintenance/remote_python.py`，显式指定`--host`和`--python`；不在外层持续新增按算子／日期命名的控制脚本。该工具不会自动选择实验、审查候选或启动模型。

从PowerShell传输含中文脚本时，优先传UTF-8文件路径，避免默认管道编码将中文变成问号。固定方法快照不可随工具更新而改写；解析修正及重新解析日志另存。本轮证据集中在`benchmarks/skill-validation/20261110-regions/`，日常调用仍使用上面的统一入口。

## 仓库内

    Ascend-Optimization/
    ├── skills/hgkda/          skill 说明与可调用入口
    ├── src/hgkda/             独立分析核心、分类知识及后端适配
    ├── schemas/              面向模型和网页的 JSON 合同
    ├── examples/             可离线运行的真实证据及 KGS 完整示例
    ├── benchmarks/           三十项算子快照与本轮验证记录
    ├── tests/                自动测试
    ├── tools/                维护、参考构建及服务器集成验证
    │   ├── skill_validation/ 通用动作入口与历史上下文兼容
    │   ├── operators/        按算子集中维护专项检查与参考适配
    │   └── archive/          退出当前入口的历史候选构建／调度脚本
    ├── docs/                 当前工程指引与专题报告；archive/为已完成旧计划
    ├── manifests/            固定依赖、来源和迁移清单
    ├── sources/              可选 KG/KGS 固定源码及原始分发包
    └── third_party/FlagGems/  固定 FlagGems 快照

Python 核心安装只使用 src/hgkda/，不会顺带安装 KG/KGS、FlagGems、CANN 或模型客户端。sources/ 和 third_party/ 保留用于开发与历史复现，版本以 manifests/ 的锁定清单为准。

`benchmarks/current.json`是当前状态索引，`index.json`是拆分前冻结索引；`operators/`、`regressions/`、`skill-validation/`及`lineage/`保存源码、合同、正式结果与生成证据。里面的方法快照不是活跃skill副本，不能在归档中开发或修改原始提示词／日志。

`build/`、`dist/`、`*.egg-info/`和Python缓存为可再生产物，不作为源码或操作入口，已由Git忽略。清理打包产物不会删除`src/hgkda/`或虚拟环境中的已安装包。

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

仓库内不再保留旧project/Ascend-Optimization-v1活动入口。历史目录可从归档或固定标签恢复；旧优化控制器保持停用。后续固定候选回归及SELU／log10实验使用独立入口并已结束，不能用旧控制器的暂停状态描述当前实验进展。服务器实际进程状态应在新任务开始前单独核对；目录说明不替代实时检查。

## 调用

Windows 使用已安装 hgkda 的 Python；910B 可直接运行：

    /data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python -m hgkda doctor

模型宿主读取 skills/hgkda/SKILL.md，并调用 scripts/run_hgkda.py 或同一套 hgkda CLI。知识随 Python 包安装；skill 不依赖个人绝对路径或旧 project 目录。

Windows现有安装环境可运行：

    .\.control-state\release-venv\Scripts\python.exe -m hgkda doctor

以上Windows命令从外层`kernelgen/`执行。从仓库根目录执行`python -m tools.skill_validation --help`查看服务器验证动作；参数与操作边界见[验证工具说明](../tools/skill_validation/README.md)。
