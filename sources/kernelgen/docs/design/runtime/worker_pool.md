# 公共 Coder 租约

`framework/worker_pool.py` 承担原 CLI 的加权 Coder 名额管理，包括 `WorkerLeasePool`、`LeaseRecord`、endpoint normalization 和 `run.max-workers` 配置读写。CLI runner 与需要申请名额的 Workflow 可使用同一实现；不在公共层反向导入 CLI。

租约是进程持有名额的记录，不是新增限制或按时间失效的 TTL。SimpleOpt 占 1 个名额，KernelGen 占 `n_parallel` 个名额；名额不足则等待，原有持有方退出时释放，失效记录通过进程启动身份检查回收。此资源与 KGS 请求线程数、物理设备 slot 独立。

## 迁移边界

原 CLI 的状态路径、原子 JSON 写入、文件锁和 KGS 进程身份检查委托移到 `framework/local_state.py`；CLI 和生命周期 workspace 独占检查共同使用这些原有 primitive，不复制锁实现。`run_control.py` 仍只承担运行控制职责，不吸收租约管理。

保留 `KERNELGEN_CLI_HOME` 和当前目录 `.kernelgen/` 的路径选择规则、`config.json`、`worker-leases.json` 与锁文件名及 Schema。不同启动目录之间不新增共享，不改变额度默认值、endpoint 归一化、锁顺序、名额申请层级、取消安全点或轮询间隔。

`cli.state` 的原有公共路径和 `cli.models.LeaseRecord` 保留为指向同一实现的导入别名，供既有调用方迁移；不保留两套实现。CLI runner/config 命令已改用公共模块。独立开发的 Catalog 业务分支在公共改动合入后更新导入，不新增第二次租约申请。

## 验证

`tests/test_worker_pool.py` 使用临时状态目录，验证公共模块可在禁止导入 CLI 的新进程中加载、旧导入别名一致、原有路径与持久化格式不变、加权跨进程占用、取消等待、PID reuse 防护、异常进程退出回收，以及 runner 正常、异常和取消路径下的单次占用与释放。

```bash
python3 -m pytest -q tests/test_worker_pool.py tests/test_cli.py tests/test_cli_batch.py tests/test_cli_history.py tests/test_cli_lifecycle.py tests/test_operator_lifecycle.py tests/test_cli_server.py tests/test_run_control.py
```

独立 worktree 测试先核对 KG/KGS 导入路径，KGS 使用 KG 锁定的 commit。不使用真实 `.kernelgen` 状态，不启动设备服务，不安装依赖。
