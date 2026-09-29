# KGS SSH stdio HTTP 代理脚本

这里只代理远端 KGS，不代理 KG Web service；健康检查使用 /status，不是 /api/runs 或 /api/config。命名实例优先使用 kg server start 管理 KGS 与代理，完整配置见 [远端部署](../../docs/operations/deployment/remote_server.md)。

## Fleet Daemon

Fleet daemon 扫描 inventory 中指定容器里的实际 KGS 端口，并为每个已发现 endpoint 维护一个固定本地端口的代理。它以原子方式写入 registry，供 Web UI 展示同一 hardware 上不同版本或端口的 KGS 实例。

```bash
./scripts/proxy/fleet.sh start
./scripts/proxy/fleet.sh status
./scripts/proxy/remote_ps_all.sh --json  # 只扫描，不修改代理
./scripts/proxy/fleet.sh stop
```

运行时状态默认存放在 `~/.kernelgen/fleet/`。托管的本地端口范围默认为 `18100-18999`，可通过 `KG_FLEET_LOCAL_PORT_START` / `KG_FLEET_LOCAL_PORT_END` 调整。默认每 60 秒扫描一次，最多并发三个 SSH session；可通过 `KG_FLEET_SCAN_INTERVAL` 和 `KG_FLEET_MAX_PARALLEL` 调整。

不要同时运行 `start_all.sh` 和 Fleet daemon。前者按 inventory 的静态端口为每个设备创建另一套代理和 worker-pool URL，Fleet daemon 则使用实际发现的 endpoint 端口。启动 Fleet 前应先通过 `./scripts/proxy/stop_all.sh` 停止旧代理。

## 推荐入口：为已有 KGS 建立单设备代理

在 KG 根目录运行；示例本地端口自行确认空闲，目标连接与远端端口读取 inventory：

```bash
bash scripts/remote_server/run_persistent_remote_http_proxy.sh \
  --device ascend --listen-port 19606
curl --noproxy '*' -fsS http://127.0.0.1:19606/status
```

这只建立代理，不启动、重启或停止远端 KGS，也不准备镜像或分配设备。停止前台脚本即断开代理。不要同时为同一个命名实例重复启动代理。

## 批量包装脚本的端口规则

start_all.sh 读取 [机器清单](../../tests/hosts.md) 的 server_port；默认本地端口计算为 18000 + 远端端口末两位，--local-port-base 可以改基数。例如远端 18306 对应默认本地 18006，与单设备示例的 19606 不矛盾。这里只说明算法，不复制一份易过时的逐设备端口表。

该包装脚本没有接管 KGS 生命周期，也不查询设备是否获准运行；不要默认启动全部设备，尤其不能据此启动昆仑芯批量优化。PID/日志位于仓库根 .proxy_pids/，与 kg server 的 .kernelgen/servers/ 分开管理；stop_all.sh 只处理前者。

当前包装脚本有待单独修复的 shell 问题：在 set -e 下首次 ((started++)) 会返回非零；08/09 端口后缀还会被 shell 算术解释成非法八进制。本次只整理文档，不宣称批量包装脚本已可稳定使用；修复前使用上面的单设备入口或 kg 命名实例。

## 行为与边界

底层长期代理用一条 SSH 会话复用多个逻辑 HTTP 流，脚本默认 max-streams=16，可显式调整；kg server 的默认值另为 KGS max-workers + 2。它们不是 Coder 数或设备 slot 数。

SSH 断开时在途请求失败，不自动重放 Eval/Profile/Debug；之后的新请求才尝试重建连接。不要仅凭代理 PID 存在就认为 KGS 健康，必须查询 /status 并检查 scheduler。

本地需要项目要求的 Python 3.10+、可用 SSH 认证和目标容器访问权限。inventory 不含私钥正文，也不保证新 clone 已配好认证。具体机器侧部署 key、代理文件、容器挂载和环境边界只由 [部署指南](../../docs/operations/deployment/remote_server.md) 维护。

连接失败先检查本地端口占用、代理日志及正确的 KGS /status；使用 curl --noproxy '*' 排除全局 HTTP proxy 劫持。不要用直接暴露远端 HTTP 端口或改成 ssh -L 绕过项目约束。
