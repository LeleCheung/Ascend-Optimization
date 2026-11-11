# 算子交付与版本

[index.json](index.json)列出30项主线与9项历史记录；当前主线26项达标。按该索引进入各算子的manifest.json，再由selected字段定位当前选用源码。未闭环候选不作为已验收交付。

每个算子按三类保存：historical_noncompliant（历史不符合当前约束）、historical_strict（历史严格Triton）、independent_strict（独立入口严格Triton）。第三类区分retained与generated：既有候选回归不等于模型重新生成。缺失类别明确记录，不补造源码或成绩。

- versions：候选原字节及绑定的审查；selected指向明确选用的版本。
- contracts：正式正确性／计时用例、参考与精度要求。relocation.json记录旧合同包和参考路径调整。
- references：PyTorch／CANN补测参考，必要原生库随目录提供。
- checks：算子补充检查与专项检查；这些是外部验收能力，不依赖development。
- evidence：选定版本的完整结果与必要来源记录；原始请求中的历史绝对路径仅供追溯，新运行由公开prepare入口生成配置。

CANN原生二进制绑定[原评测环境](environment.lock.yaml)，不能假定跨Python ABI或CANN版本通用。路径迁移保持候选、输入用例及容差不变；新参考目录重新打包会产生新包哈希，旧成绩仍绑定旧合同。运行和重新测量须采用新输出目录。

调用与配置见[公共运行接口](../hgkda/docs/runtime.md)。无需下载内部完整实验档案即可使用公开包及所选算子；native库通过Git LFS保存。早期9项只保留历史材料，不启动新设备实验。
