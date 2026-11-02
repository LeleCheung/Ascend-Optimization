# 执行方法快照

本目录保存实际加载过的控制器源码。`analysis/<算子>/controller-boundary-*.json`记录加载边界、控制器SHA和工作区状态；后续边界在`by-sha/`按内容SHA保存各工具，并保留被更新的边界记录。

`62b5c16b0892d6f6adbadc45a5115e5b8ce72ea397a4f9e7f4f000f9b37b72ee/run.py`对应hardshrink_backward开始前加载的控制器。该文件根据已提交源码撤销之后的已知修改恢复，恢复字节的SHA与当时边界记录完全一致，作为实际执行版本归档。当前运行入口仍是`tools/profiling/hgkda/strict_night/run.py`。

这些是历史执行证据；重跑使用当前入口和严格Triton规则，不能把历史快照中的旧修复或旧行为当作当前运行建议。
