# 2026-11-23 工作区整理与同步

当前软件、GQA实验和汇报资料按既有原件集中提交，不重新运行算子、不修改历史成绩。

- Excel 维护入口：[development/data](../../data/README.md)。4份原件迁移后 SHA256 一致。
- Windows、GitHub main 和 910B 工作树同步；远端独有状态先备份。
- HG-KDA 核心软件测试：101/101 通过。首次运行因 src 未在 Python path 内导入失败，加入 hgkda/src 后通过。
- research 测试：Windows 运行56项，6项 error。1项 Linux fcntl 导入不支持；其余5项是旧测试夹具缺少 compiler-verification.json 或 context.json。未据此改动实验源码或成绩。
- 提交前扫描17,883个候选文件及归档内54,255个文本条目，已配置凭据特征无命中。
- 机器地址和SSH配置留在外部配置，原始未跟踪证据和活动运行不因同步清除。

仓库外的 0_HGKDA_docs/BAAI-reports/06_work 保存最终演讲者与提交版。后续新增研究页单独设计并同步一份到仓库报告目录。
