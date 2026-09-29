# KernelGen

KernelGen 使用 AI Agent 在目标 GPU/NPU 上生成、验证和优化 Kernel。生成代码通过
独立的 KernelGen Server 执行，权威评测结果记录在 workspace ledger 中。

统一入口是[项目上手指南](docs/ONBOARDING.md)：项目介绍、安装、最小 Example 和按问题查阅文档的导航都在这里。

常用文档：

- [设计文档](DESIGN.md)
- [远端 KGS 部署](docs/operations/deployment/remote_server.md)
- [多设备实验与 E2E 验收手册](docs/operations/multiple_device_experiment_runbook.md)
- [多芯片 E2E 验证报告](docs/validation/multiple_device_e2e_validation_report.md)
- [新人培训（飞书）](https://jwolpxeehx.feishu.cn/wiki/V8vUwxGyCi4qZik0XDGchVAvnvg)

最小安装：

```bash
python3 -m pip install -e .
```

开发测试：

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q tests
```
