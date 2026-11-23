# 已验收GQA实现：固定063＋概率前瞻IR005

目标case4完整调用120.9111μs，同轮CANN127.9771μs、1.05843963×；官方case1/2/4/7通过。source.py为模型作者的冻结063源，target.hivm.mlir为CC/DeepSeek实际修改的最终IR，kernel.o是实际评测/重放绑定的二进制。

该成绩由源码和IR组合产生。运行使用`runtime-manifest-original.json`记录的原远端路径与IR注入；独立运行source.py属于源码版本。此目录是可审阅的精确产物快照，完整工具链和原环境仍在baai-910b的tle_yy、CANN9.0.0。原manifest中的绝对路径保留，复制产物未改写原验收身份。

原远端运行入口：`/data/hanle/ascend-optimization/runs/gqa-goal-20261121/controller-tools/gqa_ir_bench_driver.py`，使用`gqa-ir-qk-receive-prefetch/validation-ir-005/manifest.json`，以及原gqa/tools/attention_bench.py、gqa/contract.json和manifest中的source。复用原warmup5/trials3/iterations20、seed1729+case_id、原checker/golden与完整调用口径。每次运行输出使用新目录，历史正式结果保持冻结。

完整20项验证是独立后续范围；固定原值域[-1,1]以及既有FP16中间交接数值范围限制保留。独立kernel候选127.162544μs、CANN125.702515μs，完整调用超过1的结论与kernel-only分别记录。
