# 文档维护约定

用户入口只有 [ONBOARDING](ONBOARDING.md)：项目作用、安装、最小 Example 和按问题导航。不要再维护一份重复总索引。

## 分类

| 目录 | 内容 |
| --- | --- |
| guides/ | 用户使用指南 |
| operations/deployment/ | 当前部署、连接与环境边界 |
| operations/experiments/ | 专用实验操作和结果整理流程 |
| operations/troubleshooting/ | 事故原因及排查证据，标注发生版本 |
| validation/ | 带日期、commit、环境和范围的验证记录 |
| design/workflows/、design/runtime/、design/knowledge/ | 对应领域的契约与设计；规划能力明确标注 |
| design/ 根目录 | CLI、协议、Profiling 与跨领域设计 |
| releases/ | 已发布版本的不可覆盖记录 |
| archive/ | 仍有独有价值的历史方案与环境参考，不是部署入口 |

## 唯一事实来源

机器连接看 tests/hosts.md 的配置行，命名实例按已保存配置运行，配置变更不等于现场更新。实际设备、backend、timing、scheduler 和能力来自目标 KGS /status。当前 KGS exact commit 只从 deployment/kgs.lock.yaml 读取；历史报告只说明当时验证了什么，不反向决定今天安装哪个版本。

端口区分 SSH 入口、远端 KGS loopback 和本地 proxy；不在多个操作文档复制地址/容器/镜像表。镜像与驱动是否配套需目标验证，本地不推断，未经核验只能称为参考或历史快照。

## 更新和删除

文档与代码一起更新契约；操作步骤放操作指南，设计解释职责与边界，验证结果放 validation。不要把历史日志不断追加到现行设计，造成“尚未实现”和“已实现”同时出现。规划中的 Web、多环境 worker、硬件中立完整 case 枚举不能因有设计文档就写成已交付。

已被现行文档覆盖且没有独有证据的文件直接删除，不保留仅作跳转的旧文件；独有实验/事故记录保留并明确时点。移动文档时更新相对链接和仓库内引用，历史运行产物的绝对路径不改写。

## 本轮整理边界

最初基于 KG dev@df319f3a 的整理未连接远端或重新验证设备。后续机器清单与原厂商镜像表已统一到 [tests/hosts.md](../tests/hosts.md)，其中记录新的核验范围和历史模板边界。已删除旧 KernelGen 2.2 汇报及固定 TCP 转发入口申请，分别由现行 DESIGN.md 与 SSH stdio 部署指南替代，均可从 Git 历史恢复。

KGS 仓库未在本分支修改。其部署教程仍有旧 PPU“不再嵌套 Docker”、Agent 必须与 Server 共用 Python、旧 KG 文档路径等表述；跨仓后续修订前，KG 远端操作遵循本仓部署指南，KGS 教程只用于配套版本的服务端参数和自测。不把这些文档差异解释为需要更改实际容器或运行时。

本轮 host 验证：报告生成一致性、CLI Server、安装工具和 SSH mux proxy 共 92 passed；先核对导入来自本 worktree 和配套 KGS。机器清单的 9 条连接记录逐行一致；相对链接检查未新增断链，历史本地 runs/ 产物与跨仓文档不要求存在于独立 worktree。未遍历修复 vendored KB corpus 的上游链接，也未进行实时网络或镜像可用性验证。
