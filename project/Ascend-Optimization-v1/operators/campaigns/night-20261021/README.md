# 早期启动与传输失败记录

本目录保存正式 `night-20261021-v3` 之前的失败尝试，不作为算子性能结论。初次 pilot 暴露了服务导入路径没有绑定本仓库固定 FlagGems 的问题；后续队列又在派发前因 plan 中的 Windows 反斜线找不到源码而失败，队列被中断。

原 `queue-state.json` 仍保留中断时的 `running` 状态，没有伪改为成功。另存的 [interruption.json](interruption.json) 记录审计时原队列进程、tmux 均已不存在，且没有未回答的评测请求；`startup-failure/` 保存相关原始日志。

首次上传还发生了15个文本文件的 CRLF→LF 转换。当前目录保存910B实际文件字节，上传前的Windows字节保存在 `startup-local-before-upload/`，两者SHA对应见 [source-transfer-audit.json](source-transfer-audit.json)。这些旧合同仅用于定位失败，不能使用其历史校验值作为正式可运行合同。

有效体检与优化结果见相邻的 `night-20261021-v3/`；接口补测和额外语义修复各用独立批次保存。
