# 控制端工具

`control.py` 使用 Python 3.11 标准库和 OpenSSH，可在 Windows、Linux 上运行。默认项目路径从脚本位置定位，不依赖当前工作目录。它替代旧 `.goal-transfer/` 中可复用的 SSH 调用、上传、状态与证据下载功能。

下面从仓库根目录运行。`root` 是远端实验目录，内部包含 `Ascend-Optimization/`、`kg-runs/`、`campaign-logs/` 和 `diagnostics/`；它不是控制器 venv 路径。当前批次示例为：

```bash
ctl=project/Ascend-Optimization-v1/tools/control/control.py
root=/data/hanle/ascend-optimization/goal-20261010
```

Windows 可将 Bash 变量换成 PowerShell 变量，并使用 `python`；参数和 Python 工具相同。

## 检查状态

```bash
python3 "$ctl" --host baai-910b --remote-root "$root" status \
 --batch 20261010 --ports 19655 19657
```

输出该批次的 tmux 会话、KG 进度、完整评测结果和 KGS 队列。仅查询状态，不启动或停止进程。支持用 `--operators amin` 等参数缩小范围；当前支持三个主线算子。

## 执行 Bash 文件

```bash
python3 "$ctl" --host baai-910b exec /本地路径/check.sh
python3 "$ctl" --host baai-910b exec /本地路径/check.sh --container tle_yy
```

本地脚本通过 SSH 标准输入传输，自动去除 UTF-8 BOM、将 CRLF 转为 LF。工具确认目标 Bash 已启动，并在命令结束后输出日志；堡垒机即使返回零退出码，也不会被误判为执行成功。命令按脚本内容执行；需要持续运行的任务由远端脚本使用 tmux 托管。该入口不解析或修正脚本中的历史绝对路径。

## 上传明确的项目文件

```bash
python3 "$ctl" --host baai-910b --remote-root "$root" upload \
 operators/amin/scripts/probe-native-reduction.py
```

路径相对 `project/Ascend-Optimization-v1/`。保持源文件字节，远端校验 SHA；已有不同内容先备份到实验目录的 `script-backups/control/`。上传适用于明确的开发文件，不会提交 Git 或更新远端分支。仓库级同步使用 Git。

## 下载实验原始证据

```bash
python3 "$ctl" --host baai-910b --remote-root "$root" download \
 --batch 20261010 --operators amin matmul_bias_activation narrow_copy
```

下载指定批次的候选、报告和已成功完成的独立诊断。不下载 controller venv、Claude 二进制、Git 对象或容器层。单批原始文件总量限 256 MiB。下载完整检查通过后才写入；路径越界、符号链接、重复条目或原始证据字节冲突都会停止，不能覆盖冲突来伪造一致。

派生汇总因控制端路径而不同，可明确使用 `--replace-derived`；工具仅允许替换该批次 `master-closure-<批次>/` 下的版本对比、最终候选、精简汇报及真机诊断，旧字节先保存到仓库本地 `.control-state/download-backups/`。原始评测与附件始终拒绝覆盖。若本地仅有 CRLF/LF 差异，且远端字节与本地已提交 Git blob 完全一致，则视为已有内容，保留本地字节；不根据换行相似度直接改写证据。

下载后运行 [离线核验](../evaluation/复现与交接-20261010.md)，再审阅和提交。本工具不停止卡住的任务、不修改镜像、不自动启动实验。

## 同步 Git

正式文件提交、推送后，在远端已检查干净的工作树执行 `git fetch origin` 和 `git merge --ff-only origin/main`，随后 `git lfs pull`。远端可以继续使用 `goal-20261010` 等实验分支，不要求改名。

工作树有本机修改时先审阅、下载或备份，不执行 reset/强制覆盖。旧 `sync-final-checkout.py` 是绑定本轮工作树的恢复脚本，未作为日常控制入口保留。

完整迁移步骤见 [Linux 控制端迁移](../../../../docs/Linux控制端迁移.md)。
