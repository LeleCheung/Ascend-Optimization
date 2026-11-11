# 内部方法验证工具

本目录维护prepare-history、audit、finalize、export、verify。它们负责上游起点实验、方法快照、生成来源和档案核验，不是对外skill运行依赖。

设备采集、上下文、模型调用、候选预检与正式评测已提取至已安装的hgkda.runtime。执行python -m hgkda.runtime --help查看公共动作。

历史experiment快照仍保留旧路径及SHA，不能改写原件。迁移映射见development/manifests/three-part-migration-20261111.json。
