# Benchmark core reference 验证

`POST /reference` 使用已安装 Catalog 或 Bundle binding，接受 `benchmark_fingerprint` 和 `settings.timeout_seconds`，不接受 candidate/implementation。`/status.capabilities.benchmark_reference` 声明 `enabled=true`、`evaluators=["flaggems"]`、`level=core`；这是 KGS 接口能力，目标 Gems 仍须支持 benchmark `--reference-only`，否则返回 `UNSUPPORTED`，不通过 release 或 commit 猜测功能。

KGS 使用原 Gems benchmark suite，调用 `--reference-only` 并使用该模式默认的 core level，不再显式传 `--level core`；不运行 correctness pytest，不做候选注入、warmup、计时或 Profile。必须使用 core 的 Catalog binding，核对请求冻结的 benchmark fingerprint，并验证成功报告覆盖 inspect 返回的整个 core case 集合。重复、缺失、外来 case 或未真实调用不能记为通过。

返回 `ReferenceResult`：`scope=benchmark_core`、结构化 status、原始 report、fingerprint 和 log。`PASSED` 仅表示原性能 baseline 可执行，不证明 correctness reference、数学语义或有效计时。`ALL_SKIP`、`NO_CASES`、`UNSUPPORTED` 与失败单独保存，不能只看 pytest exit code 或 HTTP 200 判断就绪。

该操作复用 EvaluationExecutor、设备 slot 和非 daemon 隔离 worker。`operation_cancel.operations` 包含 `reference`，支持相同的 operation ID 登记、排队取消和运行中取消；超时或中断后按原规则强探针恢复，不换卡重试。请求审计保留 binding 和 request SHA，但不伪造 implementation/candidate SHA。

SDK：`kernelgen_server.client.reference(ReferenceRequest(...), server_url, operation_id=...)`。Native Catalog 继续使用原 reference-as-solution Preflight/Eval 路径，不要求其模拟 Gems benchmark 协议。
