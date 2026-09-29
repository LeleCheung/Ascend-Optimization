# 候选环境副作用准入验证（2026-09-08）

> 历史证据：本文的版本、地址、端口、容器、镜像及设备占用仅代表当次实验，不是当前部署配置。新实验按 [部署指南](../operations/deployment/remote_server.md) 与机器清单准备；不要直接重放历史启动命令。

本轮在 KG `feat/candidate-admission` 同步 Coder prompt，在 KGS `feat/metadata-admission-policy` 补齐静态规则。未合入主线、未发布或部署；Protocol v6.2 不变。只修改候选源码准入，不改变目标 smoke、精度与计时实现。

## 检查与结果

KGS admission、metadata、FlagGems adapter 与既有 hack 检测 host 测试 101 项通过。实际被测模块从对应 Server worktree 导入；adapter 哨兵验证新增拒绝发生在 candidate import/pytest 之前。KG Coder/Analyzer/完成状态相关测试 59 项通过，被测 implementation profile 从对应 KG worktree 导入。未新增 GPU 运行，也没有安装依赖或改变 Torch/Triton/厂商运行时。

历史回放扫描 348 份候选，21 条缺源码，所有已扫描候选 SHA 均与归档吻合。Gems 模式拒绝数从 20 增至 30；新增 10 份如下。native 同时回放，Gems 注册 context 的专属规则只在 Gems adapter 拒绝。

| 厂商/行 | 算子 | 新拒绝依据 |
|---|---|---|
| tianshu:5 | `concatenate` | candidate environment mutation: candidate.py:11 torch.cuda.memory._set_allocator_settings |
| tianshu:6 | `mvlgamma_` | candidate environment mutation: candidate.py:67 torch.cuda.empty_cache |
| tianshu:8 | `index_copy_` | candidate environment mutation: candidate.py:420 torch.cuda.empty_cache |
| tianshu:33 | `special_bessel_j0` | candidate environment mutation: candidate.py:121 torch.cuda.empty_cache |
| tianshu:34 | `special_bessel_y0` | candidate environment mutation: candidate.py:117 torch.cuda.empty_cache；candidate environment mutation: candidate.py:84 torch.cuda.empty_cache |
| tianshu:52 | `mvlgamma` | candidate environment mutation: candidate.py:106 torch.cuda.empty_cache；candidate environment mutation: candidate.py:133 torch.cuda.empty_cache |
| tianshu:59 | `special_multigammaln` | candidate environment mutation: candidate.py:180 torch.cuda.empty_cache；candidate environment mutation: candidate.py:211 torch.cuda.empty_cache |
| pingtouge:21 | `float_power_` | protected candidate state: candidate.py:378 flag_gems.testing.override_registered_op |
| pingtouge:41 | `unbind_copy` | candidate environment mutation: candidate.py:74 open in write mode；candidate environment mutation: candidate.py:76 os.chmod；candidate environment mutation: candidate.py:77 os.environ；candidate environment mutation: candidate.py:78 os.environ |
| muxi:34 | `linear` | candidate environment mutation: candidate.py:12 os.environ.pop；candidate environment mutation: candidate.py:84 os.environ.pop |

旧标签中的 115 份“正常对照”有 5 份也被新增规则拒绝：tianshu:5, tianshu:8, tianshu:33, tianshu:52, tianshu:59。它们确实含 empty_cache 或 allocator 修改，属于新策略明确禁止的源码行为；其余 110 份通过。正常标签只表示当时测试未观察到失败，不是源码无副作用证明；保留原标签，不声称“115 份零误杀”。这不证明它们是原表降速原因，也不扩大原表失败统计。

明确文件写入或外部进程调用在候选源码内拒绝，避免按编译器 shim 的特定文件名匹配；读文件、读取环境、修改环境副本、元数据查询、正常张量分配及无法确定类型的同名普通方法不据此拒绝。规则不构成任意 Python 沙箱，动态拼接、间接调用和未识别模式不具备完整保证。

policy_sha256：`bd6a7bc3f46a71e6d1b90223e911ad2ea463af4b314348de755177f46963d1dc`；capability version=1、stages=[preflight]。源码规则变更由既有规则 hash 使旧 receipt 失效，不引入后续重复扫描。

## 本地证据

`/data/akg_kernel_bench_lite/kernelgen/runs/preflight-environment-policy-20260908/` 保存 `scan.py`、`generic.json` 和 `summary.json`；输入位于授权实验归档，不随新 clone 分发。恢复输入后在 KGS worktree 执行：

```bash
PYTHONPATH=. python3 /data/akg_kernel_bench_lite/kernelgen/runs/preflight-environment-policy-20260908/scan.py
```
