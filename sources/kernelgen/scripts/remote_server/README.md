# 远端 KernelGen Server 代理脚本

本目录只保存运行时入口和实现；对应 host 测试位于 `tests/test_ssh_stdio_http_proxy.py` 和 `tests/test_ssh_stdio_http_mux_proxy.py`。

正式远端实验使用 `run_persistent_remote_http_proxy.sh`。它通过 `ssh_stdio_http_mux_proxy.py` 为每个目标维护一条长期 SSH 会话，并在会话内多路复用 HTTP 连接。`run_remote_http_proxy.sh` 与 `ssh_stdio_http_proxy.py` 是每个本地连接新建 SSH 的兼容实现，不用于高并发实验。

`kg server start <name> --target remote --ssh-command ... --container ...` 直接复用 `ssh_stdio_http_mux_proxy.py`。CLI 以命名实例负责远端 KGS 和本地代理的成对启停及代理恢复；Shell 入口继续用于基于 inventory 的多设备实验。

两个 Shell 入口默认读取 `tests/hosts.md`，也可用 `--inventory` 指定其他清单。完整部署和运行步骤见 `docs/ONBOARDING.md`、`docs/operations/multiple_device_experiment_runbook.md` 和 KernelGen Server 仓库的 `docs/multiple_device_deploy_tutorial.md`。
