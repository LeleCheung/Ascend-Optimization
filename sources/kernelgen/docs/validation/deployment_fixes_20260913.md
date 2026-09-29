# KG v6.3.0 部署问题修复与复验

本记录对应 [发布版天数验收](release_v6.3.0_acceptance_20260913.md) 发现的四项问题。修复最初分别保存在独立分支，不把修复分支测试结果归入原版通过范围。用户随后授权全部 PR 合入并重发最终 KG v6.3.0；本次合并回归与 tag 替换记录见 [发布说明](../releases/v6.3.0.md)，KGS 修复按 patch 发布为 v6.3.3。

| 仓库与分支 | 提交 | 修复与验证 |
| --- | --- | --- |
| KG `fix/proxy-start-readiness` | `7bd309d8`、`b36ba723` | 先绑定本地端口，再建立 SSH；用子进程专属一次性管道通知父进程。失败前保留本次远端启动身份，保证后续 stop 仍使用正确 PID 校验。代理与 Server 相关 77 项 host 测试通过 |
| KG `fix/server-python-venv` | `234c6d8a` | 保留解释器 symlink 路径，不解析到系统 Python；默认 symlink/copy venv、PATH 与显式路径测试，共 61 项通过 |
| KG `fix/kgs-install-bootstrap` | `cdd154fa` | 恢复 pip 构建隔离，构建依赖以 KGS pyproject 为准；保留脱敏限长错误摘要。67 项 host 测试通过 |
| KG `fix/remote-doctor-boundary` | `22760ca3` | remote doctor 不检查本地 checkout；本地 doctor 按实例记录的 commit 校验。62 项 host 测试通过 |
| KGS `fix/management-build-isolation` | `c70c18a` | 远端管理安装路径同样恢复构建隔离；12 项 management 测试通过 |

各行测试有重叠，不相加为独立测试总数。所有 KG 测试都通过临时父目录的 kernelgen 链接隔离，并核对模块导入路径。KGS worktree 改为合法包名的目录后运行测试；最初非标准目录名导致 pytest 导入根 shim 报错，不归为实现失败。

## 干净环境安装

真实 Python 3.12 默认 venv，初始没有 setuptools 和 wheel。使用 KG 安装修复分支的 `_install_server` 安装 KGS 修复 checkout，成功；随后导入位置检查、重复安装与 `pip check` 通过，重复安装前后记录的包版本一致。证据为本地 `/tmp/kg-install-repair-r9V8uV/result.json`，没有安装 Torch/Triton/厂商运行时。新增轻量运行包与精确版本见结果文件，构建工具由 pip 隔离环境按 KGS 声明准备。

错误摘要测试覆盖缺少 setuptools 的原始错误、URL/环境凭据/Authorization/带引号密码的脱敏、长度限制，以及私钥标记时不输出正文。测试中的凭据字符串与 example.test URL 均为合成 fixture，不是真实配置。

## 天数部署复验

复用本轮授权的共享物理卡 15、现有容器及本次独立实例 `tianshu-v630-jouxh3`。KG 通过固定 worktree 的 PYTHONPATH 运行，配套远端仍为 lock 中的 KGS v6.3.2@ca456d64deb30a5b90eafdebb1845c5feed180d7；没有将尚未合入的 KGS 修复直接覆盖到远端。此轮 KGS 安装修复验证范围为 host 与真实干净 CPU venv；后续已合入 KGS !51 并发布 v6.3.3，最终 KG lock 已更新。本节远端样本仍为 v6.3.2，不改写为新版本实测。

修复后的代理在本地 loopback 21631 启动，监听 PID 284775 与受管记录一致；远端 KGS PID 3745522 只监听 loopback 18301，slot=1、请求 workers=2。重复 start 返回 ALREADY_RUNNING 且 PID 不变。独立 doctor 修复分支查询同一实例返回 OK，不需要本地部署 checkout。经新代理运行真实 reference Preflight/Eval、Triton candidate 和并发评测通过，完整结果为发布验收证据根中的 `fixed-proxy-server-smoke.json`（passed=true）；本次没有重复 resilience 故障注入或模型优化全流程。

随后停止本次实例，将其本地端口临时配置为既有代理占用的 19601。新代理 bind 失败时，CLI 正确返回 exit code 2 和 readiness 错误，没有借旧代理的正常响应报告 RUNNING。收尾发现旧进程记录仍指向上次远端 KGS，stop 被 PID 校验拒绝；补充提交 `b36ba723` 将本次 remote start 返回身份提前保存，复测相同端口冲突后 stop 成功，不取消 PID reuse 防护。恢复实例配置为 21631 并保持停止。

本次仅关闭自己的临时 KGS 和受管代理，21631 无监听；旧代理 PID 3615951 始终未受影响，其他容器服务没有停止或修改。没有新增模型任务，也没有变更远端 Torch、Triton 或厂商运行时。

## 合入顺序

四个 KG 修复及文档已分别经 !115～!119 合入；安装主题的 KGS !51 独立合入，!52 发布为 v6.3.3。最终 KG v6.3.0 更新 lock、推荐组合和发布记录。KG tag 按用户本次明确授权替换，KGS v6.3.2 tag 不动；不把旧版本完整 E2E 冒充最终组合的完整 E2E。
