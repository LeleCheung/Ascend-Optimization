# HG-KDA / Ascend-Optimization

仓库按用途分为三个部分：独立性能诊断与优化工具、算子交付、开发研究。

| 目录 | 面向谁 | 入口 |
| --- | --- | --- |
| **hgkda/** | 使用或集成HG-KDA的开发者与模型宿主 | [安装与使用](hgkda/README.md)、[Skill](hgkda/skills/hgkda/SKILL.md) |
| **operators/** | 使用和复验算子实现的开发者 | [交付说明](operators/README.md)、[版本索引](operators/index.json) |
| **development/** | 维护HG-KDA、开展研究实验的人 | [开发指引](development/README.md) |

外部运行仅使用hgkda、所选算子的交付材料和显式配置的外部服务；不导入或执行development中的文件。

从仓库根目录安装：

    python -m pip install ./hgkda
    python -m hgkda doctor
    python -m hgkda.runtime --help

当前主线30项中26项按既定口径达标，另保留早期9项历史记录。整理目录不增加算子交付数量、不改变候选源码或历史成绩。版本、参考类别、合同与完整证据以operators中的清单为准。

虚拟环境、模型凭据、编译缓存和活动运行目录放在仓库外。共同维护规则见[AGENTS.md](AGENTS.md)。
