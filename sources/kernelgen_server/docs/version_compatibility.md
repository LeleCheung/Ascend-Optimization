# 版本兼容与运行选择规则

KernelGen、KernelGen Server（KGS）和 Protocol 版本相互独立。配套 KGS repository、release 和 exact commit 由 KG 的 `deployment/kgs.lock.yaml` 唯一决定。KGS 根目录的 [`compatibility.yaml`](../compatibility.yaml) 声明自身 release、支持的 Protocol/Catalog 模式和默认 framework 推荐值；其中历史 `kernelgen_releases` 不再用于反向限制新的 KG release。

## 当前兼容边界

当前发布组合为 KG v6.3.0 / KGS v6.3.2 / Protocol v6.2，发布门禁与历史 E2E 的精确范围见 [发布说明](releases/v6.3.2.md)。KGS 的 `/status.api_version` 为 `v6.2`，同时加载 `v6.0` 和 `v6.2` Catalog；compatibility.yaml 中的旧推荐矩阵仅保留历史，不随新发布反向登记 KG：

| Catalog `api_version` | 默认模式 | evaluator | layout |
|---|---|---|---|
| `v6.0` | adapter | `flaggems` | `flat` |
| `v6.2` | native | `native` | `per-operator` |

`v6.0` 表示 FlagGems adapter Catalog，不表示部署旧版 Server。Catalog 的 `api_version` 与当前 Server wire protocol 版本不能混为一谈；运行时协议兼容只看 `/status.api_version`，可选功能只看 capabilities。

KGS v6.3.1 的 FlagGems Adapter Catalog 已取消 framework 版本绑定：KG 实例选择实际 checkout，KGS 读取实际 HEAD 并纳入 benchmark fingerprint，仍校验相关源码工作树和测试资产。旧 Adapter manifest 的 framework 字段不再限制执行；FlagGems-backed native Catalog 的 exact revision 契约不变。默认 framework 推荐值保持 `YaooXu/FlagGems@d64794e63b502cb836bc015a92a62c42de4be05a`。旧 KGS v6.3.0 tag 不含此放宽且保持不变。

## 新实验如何选版本

1. 从待跑 Catalog 的 `manifest.json` 读取 `api_version` 和 `evaluator`，不得根据目录名猜测模式。`v6.0` 应为 adapter，`v6.2` 应为 native。
2. 使用当前 KG 锁定清单中的 KGS release 和 exact commit，核对 checkout 来源、HEAD、clean 状态以及 KGS 自描述；不按 Git tag 的 SemVer 排序或 KGS 历史反向矩阵自行选择。
3. 首次默认按 `frameworks.flaggems` 推荐值准备 checkout，并设置 `KGS_FLAGGEMS_ROOT`。支持上述放宽的 KGS 可对 Adapter 使用 KG 实例显式选定的其他 commit，不要求仅为 commit 变化修改 Adapter Catalog；仍需重新 inspect 和验证 baseline。Native 使用 FlagGems 时必须匹配其 Catalog 的 exact revision。每个 campaign 固定实际 commit，不自动更新。
4. checkout 对应 tag 后启动 KGS，读取 `/status`，核对 `api_version`、backend、timing、capabilities 和 scheduler。KG/KGS 软件版本通过 tag、commit 和包元数据记录，不用 release 推断 wire 兼容；清单不能替代目标芯片启动探针和 reference-as-solution 自测。
5. 将 `KG tag@commit / KGS tag@commit / Protocol api_version / Catalog api_version / evaluator / FlagGems commit` 写入实验记录。

中断续跑必须使用 workspace 已记录的 KGS release，不在 campaign 中途自动升级。若确需升级，应结束当前 campaign，保留原始证据，并以新 run name 创建新 campaign。

## 维护规则

- 发布 KGS 时，在同一 release feature 中同步软件版本、`compatibility.yaml`、相关文档和版本测试。
- KG 锁定清单是部署选择的唯一来源，`validation_scope` 必须如实反映 host 或真机验证范围；历史推荐值不等于仓库中的最大 tag。
- `frameworks.flaggems.revision_policy=exact` 规定默认部署快照，不再要求 Adapter Catalog 重复保存该版本。变更推荐值时更新兼容清单、测试和验证记录；Native Catalog 是否更新其冻结 oracle 版本须单独验证。同名 branch 的新 HEAD 不会自动改变已创建实例或 campaign。
- 新增 KG release 不要求修改旧 KGS 的 `kernelgen_releases` 或移动已有 tag；历史映射保留作追溯，不继续增长交叉版本矩阵。
- 修改 `supported_api_versions`、默认模式或 evaluator/layout 约束时，必须同时修改协议代码、Catalog 校验、迁移文档和测试。
