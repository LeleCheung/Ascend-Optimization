# 调用接口

先在项目根目录安装 python -m pip install .。skill 脚本与 python -m hgkda 使用同一入口。

    python -m hgkda doctor
    python -m hgkda compare --directory examples/paired-expm1/paired --output /tmp/comparison.json
    python -m hgkda context --operator expm1_ --source examples/paired-expm1/candidate.py --comparison /tmp/comparison.json --output /tmp/context.json
    python -m hgkda hotspots --listing examples/matmul-instructions/listing.json --source examples/matmul-instructions/candidate.py
    python -m hgkda regions --listing examples/matmul-instructions/listing.json --source-map examples/matmul-instructions/source-map.json
    python -m hgkda knowledge --operator expm1_

Windows 将 /tmp/ 换成项目外的新输出目录。每次输出使用新文件名。

hgkda.result/v1 封装含 operation、status、inputs 和 result。每个输入记录 SHA-256；结果可直接由模型或网页读取。compare 的输入是同一 spec.json 下两侧 collection.json 及 CSV。hotspots 要求指令清单声明 source_sha256；历史清单需由采集请求和源码校验后显式导入，不能仅凭文件名绑定。

context 的 facts 保存对照、源码/IR、Roofline、热点和区域；hypotheses 单独保存经验规则结果；knowledge 给出版本化参考。它不自动调用任何模型，也不宣称模型已按建议完成优化。

roofline --input <json> 接受已核验的 GEMM 工作量、同次采集指标和同环境经验参考：facts、shape（M,N,K）、dtype、empirical，并保留 source_sha256 与 case。缺少工作量合同时使用成对 profiling，不用估计值填补未知字段。

## 知识整理

curate --records <records.json> --root <evidence-root> 按算子类型、动作和作用域去重。只有源码、完整请求/结果、候选后端审查及所需补测身份均匹配，才保留 validated 状态；缺失证据降为 draft，旧条目保留在 superseded。

retrieve --index <index.json> --operator-type <type> --scope <scope.json> --max-chars 6000 重新校验证据后按作用域检索，不截断单条记录。它是确定性整理规则；模型提出的新经验不会未经验证自动成为成功规律。

gate --source <candidate.py> 只执行已知路径静态筛查；--review 可绑定已有审查记录。KGS evaluate 还需冻结请求、审查与新的输出目录。后端详细用法见仓库 docs/backends.md。
