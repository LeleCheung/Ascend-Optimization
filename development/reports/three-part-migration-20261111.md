# 三部分目录迁移与独立运行验证

日期：2026-11-11。迁移起点：8858e100ab4684f40c970f59dc5147bfd743919b。

## 目录与职责

| 目录 | 职责 | 日常入口 |
| --- | --- | --- |
| hgkda/ | 面向外部开发者的 skill、Python 包、知识和采集／模型／评测接口 | skills/hgkda/SKILL.md、python -m hgkda、python -m hgkda.runtime |
| operators/ | 算子版本、合同、PyTorch／CANN 参考、补充检查与验收证据 | index.json → 各算子 manifest.json → selected |
| development/ | HG-KDA 软件测试、研发工具、冻结实验、报告与上游快照 | README.md、tests/hgkda/、tools/ |

根 README 负责导航；hgkda/README.md 面向使用者；development/README.md 面向维护者。外部运行不导入或执行 development 中的代码。软件测试不属于 skill 正常调用；算子补充与专项检查位于 operators/<算子>/checks/，对应参考位于 references/。

## 版本与证据

- operators 保存 30 项主线及 9 项历史记录；已达标数量仍为 26/30。
- 三类为历史不符合当前约束、历史严格 Triton、独立入口严格 Triton。第三类区分既有候选回归 retained 与模型新生成 generated，不将回归表述为重新生成。
- 33,503 份冻结原件逐文件 SHA 核验通过；526 项交付清单中的源码与证据引用核验通过。缺失版本不补造。
- 候选 kernel 与启动封装保持原字节。历史请求、性能结果及来源记录不改写。
- 参考中原绝对路径改为随算子交付的原生库路径，库文件按原 SHA 核验；合同输入与容差不变。重新打包后合同哈希变化，旧成绩继续绑定旧合同。
- contracts/relocation.json 记录迁移，不放入 KGS 执行合同包。首次隔离上传因该说明文件被拒收，尚未提交完整评测；修正后使用新运行目录验证，失败记录保留在运行归档。

## 已完成的软件验证

- 56 项软件测试通过，包含仅复制 hgkda 与 operators 的隔离准备／上下文测试。
- Skill 结构检查通过；公开代码没有 development 导入或必需脚本路径。
- Windows 已安装与 910B 隔离环境相同的 wheel；不设置仓库 PYTHONPATH 时 doctor 和软件测试通过。
- 公开交付中未发现 model-settings.json、credentials.json 或 .env 凭据文件。

## 真机验证口径

使用固定 log10_ 交付源码 d5ba772425b156c2f9377b8c07ceb338c5045024f81169042d76690645c5f049，在只有 hgkda 与 operators 的分发目录调用公共入口。KGS 独立部署，使用物理卡 7 和项目锁。验证包括准备、上下文、预检、完整评测、双参考补测、特殊值检查与成对 profiling。

真机验证通过：正式 36/36，PyTorch 与 CANN 补测各 66/66，特殊值 15/15；本次相对 CANN 的完整调用加速比为 0.899286×。同输入候选／CANN 成对采集、实际 IR、指标比较和上下文生成均完成；正式性能反馈及成对证据已进入新上下文。

本轮不调用模型重新生成候选，不增加交付数量，也不覆盖原交付性能。独立 KGS 临时服务已关闭，只处理本项目进程。127 份验证材料共约 1.1 MB，逐文件哈希核验通过，见[证据目录](../experiments/migration-20261111/result.json)和[核验记录](../manifests/layout-verified-20261111.json)。

Windows／910B 使用同一 wheel，SHA256 为 4e18e0dfc7da5593c83e9251b1d291cb8d1d8710a15ff082445a63ad940449c4。分发目录没有 development，公开包与算子交付材料可以独立运行。

## 维护入口

    python -m pip install ./hgkda
    python -m hgkda doctor
    python -m hgkda.runtime --help
    python -m unittest discover -s development/tests/hgkda -v
    python development/tools/maintenance/verify_layout.py --repository . --output <新文件.json>

完整原始实验迁至 development/experiments/；按 development/manifests/three-part-migration-20261111.json 查旧路径。历史文档内的旧路径是原记录，不作为当前操作入口。运行目录、凭据、缓存与构建产物位于仓库外。
