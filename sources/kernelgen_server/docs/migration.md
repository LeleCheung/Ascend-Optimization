# V6 adapter 迁移说明

## 新定义

V6 不再等同于“用 Definition/Workload 复刻所有 pytest”。它由三部分组成：

- KernelBench、FlashInfer-bench 使用简化 Workload runner；
- 复杂且与框架耦合的评测使用 framework adapter；
- 所有 runner 统一返回 `EvaluateResponse`。

简化 runner 从上一版设计保留了严格有序 ABI、默认值、简单 mutation/alias、可信
`gen_inputs`/`correctness`/`valid` hook、PyTree compare 和目标设备计时。v6.2 的
correctness/effects gate 只作用于 correctness workload；timing workload 与 framework
benchmark 一样只负责性能测量。它删除了参数类型 DSL、Workload `call`、条件 effects、
预期异常和独立 correctness source 字段；真实 `*args` 由 `var_positional` 保留，但不
接受未解析的 `**kwargs`。

FlagGems adapter 直接运行原始 accuracy pytest 和 `--level core` benchmark，不再从
这些测试抽取 Workload。请求仅上传 binding、candidate 与执行设置；Definition 来自
受信 Catalog，测试文件和 candidate 注入由 adapter 维护。最初只用 `addmm_` 做 POC，
当前支持范围应从 `data/flaggems-adapter-definitions` 和 `/inspect` 动态读取。

## 兼容边界

KGS `v6.2.4` 对外报告 `/status.api_version=v6.2`，同时接受 `v6.0` 和 `v6.2` Catalog，但不兼容 V5。`v6.0` 默认表示 `evaluator=flaggems`、`layout=flat` 的 adapter 模式；`v6.2` 默认表示 `evaluator=native`、`layout=per-operator` 的 native 模式。运行选择和发布映射以根目录 [`compatibility.yaml`](../compatibility.yaml) 及[版本兼容与运行选择规则](version_compatibility.md)为准。

简化 runner 的最小 fixture 位于 `kernelgen_server/data/simple-v6-test`；正式 v6.2 native Catalog 位于 `data/flaggems-native` 和 `data/kernelgenbench`，FlagGems framework adapter Definition 位于 `data/flaggems-adapter-definitions`。

仓库中保留的 `v5.1` Catalog 统一归档在 `data/.old/`，不再出现在可运行 Catalog
的顶层目录，也不能交给当前 Server 执行。

## 当前规范与验证

- 新增 native Catalog 按
  [v6.2 Native Catalog 接入规范](v6.2_native_catalog_authoring.md)提供资产和证据；
- 实际芯片、软件环境、case 数、结果和已知问题只记录在
  [多芯片 Eval 验证报告](multiple_device_eval_validation_report.md)，不在迁移说明中
  复制易过期快照；
- 部署后的 reference-as-solution、preflight、容错和资源释放按
  [多芯片部署与自测教程](multiple_device_deploy_tutorial.md)执行。
