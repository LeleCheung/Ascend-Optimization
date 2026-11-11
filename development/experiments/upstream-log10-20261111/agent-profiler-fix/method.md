# FlagGems 上游验收

准备向 FlagGems 提交算子时，在 KGS 完整评测和算子补充检查之外，运行目标仓库原生 tests 与 benchmark。三层结果独立保存，不以其中一层替代其他层。

## 工作区与职责

- 使用显式配置的外部 Git 工作区，固定 upstream_commit。不要修改内部冻结快照，也不让公开运行依赖 development。
- CC／DeepSeek 负责候选的接口接入、设备分发及实现调整。Codex 可维护验收适配器、分析失败并更新方法，不能代写目标实现。
- 保留交付起点 SHA、模型输入输出、接入补丁、实际加载的模块、测试命令及原始结果。接入后的源码单独标识，不能继承旧源码的 KGS 通过结论。
- 先运行目标算子的原生正确性与性能测试，再按改动范围补关联回归和目标分支 CI。测试通过不代表 PR 已获评审通过。

## 公开入口

    python -m hgkda.runtime upstream-test --config /path/upstream-test.json

配置包含 repository、upstream_commit、operator、integration_files（源码相对路径到 SHA256 的映射）、expected_function_source、delivery_source、delivery_sha256、python、sdk、device、lock、output，以及可选 container、environment、timeout_seconds。

output 位于工作区之外，必须是新目录。模型凭据不放入本配置。仅允许声明的 src/flag_gems/ 文件发生接入修改；测试、基准、精度要求与配置保持目标 commit 原样。根据实际环境配置设备，遵守设备独占锁。

适配器验证导入来源后调用外部 tools/run_tests.py，使用 --ops 精确选择算子，不启用 --quick，保存完整日志。上游 --stages 是算子成熟度过滤，不能理解为正确性／性能阶段。当前适配器使用该仓库脚本的 core benchmark 口径，完整记录跳过项及收集错误。

默认 mode=scheduler 调用原生调度脚本。若其全仓库收集被无关测试阻断，保留失败证据，可显式选择 mode=operator-files，并提供 accuracy_files、benchmark_files（相对外部仓库的完整文件路径），可选 marker 与 accuracy_reference。此模式直接运行这些未修改的上游pytest文件，复用原生结果解析函数；不自动忽略报错模块。提供 expected_accuracy_cases_from 可核对与原调度器选中的用例集合完全一致。两个模式的结果分开记录，精确文件通过不能写成原调度器全流程通过。

## 结果解释

result.json 保留正确性计数、跳过项、逐 dtype／shape 的基准结果和派生几何平均加速比。必须有实际通过用例和有效计时；即使调度脚本返回零，缺失结果、测试收集错误或全部跳过仍不能判为成功。

benchmark 的参考与计时方式遵循目标仓库；结果单列为 FlagGems 原生 benchmark，不能覆盖固定 KGS／CANN 成绩。若测试失败，区分候选问题、接口接入问题和环境／上游测试基础设施问题，记录证据后修复，不删除失败用例或放宽容差。

## 启动接口与 profiling 兼容

普通计时通过不代表原生 profiler 计时可用。若快速启动把张量替换为整数地址，需检查编译后 launcher 在 profiler 激活时是否读取张量形状。例如在 log10_ 的上游验收中，正确性18项通过，而 profiler benchmark 在直接launch处出现 int 没有 size 属性；对应厂商launcher含 tensor.size() 调用。

诊断应对照普通调用与profiler调用的参数合同，区分设备数学错误和宿主启动／采集接口错误。由生成模型选择保持Tensor元数据的启动方式，或在需要时使用受支持的Triton启动入口；不关闭profiling、不修改计时器，也不以厂商原生计算替代候选。修改后重新验证上游benchmark，并对新源码重新执行KGS及必要补测，不能沿用旧源码的成绩。
