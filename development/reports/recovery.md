# 历史恢复与目录迁移

## 恢复点

发布整理前的源码、原始采集、模型工作区快照、逐 case 结果及暂停记录固定于：

- 提交：f63e6c30a0d8583ba092844e57ab90c0f6d0349f
- 私有远端标签：archive/pre-release-20261103
- 暂停记录：project/Ascend-Optimization-v1/operators/campaigns/strict-next-20261103/user-pause.json

该检查点已经推送并通过本地 Git LFS 对象核验。历史文件从维护目录移出，不改写原始实验，不删除 Git 历史。需要复现旧实验时，在独立目录检出标签，不覆盖当前工程：

    git worktree add --detach ../Ascend-Optimization-history archive/pre-release-20261103
    git -C ../Ascend-Optimization-history lfs pull

历史脚本包含原机器路径和环境依赖，应先阅读暂停状态及对应批次说明。不要直接运行旧 launch/prepare 脚本，不能把旧搜索分数当作重新验证结果。

## 当前可用材料

benchmarks/index.json保留拆分前三十项算子的顺序、分组、源码和核心结果，其中25/30是当时状态；当前26/30及后续变化以[当前索引](../benchmarks/current.json)为准。examples/提供独立包可使用的成对指标、指令清单与完整KGS合同。manifests/example-provenance.json和release-migration.json记录精选原件及迁移逻辑的来源。

当前 skill 不再依赖旧 project 目录。运行环境、缓存、服务日志和新实验输出放在仓库外；可选 KG/KGS 与 FlagGems 固定快照仍保留在 sources/ 和 third_party/，不因目录整理而变更版本。

## 本地清理

清理清单见 manifests/windows-cleanup.json。外层的一次性传输包及提交辅助清单完成核验后删除；六份未匹配到当前 Git 对象的历史中间版本单独保存于控制目录的 recovery/unique-transfer-files，并记录原包、路径及 SHA。

旧的仓库内.control-state移至仓库外的recovery/，保留未纳入Git的本地诊断和同步回执。旧归档不是新工作流依赖；外层.control-state/release-venv/则仍是当前Windows已安装环境，不能混同或整目录删除。用户编辑器配置保持原位且不提交。

2026-11-10的外层整理将一次性脚本、回执、传输包及实验副本归入`.control-state/archive/20261110/`，原路径、SHA及恢复位置由该目录`manifest.json`记录。归档脚本不作为入口；若需重放其绝对路径依赖，先按清单在独立环境恢复。当前可复用的远程Python入口是`tools/maintenance/remote_python.py`，主机和解释器均由参数指定。
