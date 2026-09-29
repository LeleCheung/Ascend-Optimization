# KGS 进程管理入口

`kernelgen_server/management.py ACTION BASE64_JSON` 是 KG 调用的目标侧管理命令，不是 HTTP API，也不要求预先导入 KGS 或安装 HTTP 依赖。支持 start、status、stop、logs、follow_logs、doctor 和 install_flaggems。KG 负责通过 SSH/container 命令准备并校验精确 Git checkout，随后执行该文件；不要通过 SSH 发送 KG 管理脚本源码。

start 按实际安装状态准备 KGS 的轻量依赖，记录 state_root 下的 package-changes.json，并检查 Torch、Triton 和厂商包版本未变化；没有离线或安装许可开关。环境文件仅在部署期间加载，既不打印也不注入后台 Server。进程只绑定 loopback，PID 与 /proc 启动 ticks 同时匹配才视为活动；僵尸进程不占有实例，无法读取身份则报错，不退回 PID-only 判断。

实例 state_root 保存进程身份、日志和部署证据，不绑定 Catalog。该命令与 HTTP Protocol 版本独立；KG 仍只通过 /status.api_version 和 capabilities 判断执行接口的兼容性。
