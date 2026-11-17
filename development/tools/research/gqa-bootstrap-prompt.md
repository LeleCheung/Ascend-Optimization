加载skills/hgkda/SKILL.md。本任务建立CANNBench Level 4 GQA的严格Triton起点，合同见contract.json，语义和精度见task-description.md，官方数学参考见golden.py。只读写当前工作区，不访问其他实验和历史候选，不使用其他agent。

candidate.py当前为空文件。实现公共入口run(query,key,value,scaleValue=-1.0,is_causal=False)，返回与query同shape、dtype的BSND张量。支持合同列出的全部4个官方case，不依赖case_id、输入值、测试seed或答案缓存；按真实shape/stride和参数处理。query head映射到对应KV head，保持KV共享，不物化重复KV。因果mask是右下角对齐，不能误用左上角。

设备计算只用Triton。Python仅分配、元数据和kernel启动，禁止PyTorch matmul、softmax、索引计算或原生attention等设备计算替代候选。允许torch.empty。初次实现重视精度和UB约束，先建立可正确执行的起点，再研究性能；不得放宽合同MERE/MARE阈值，附加allclose不等于正式精度达标。

可选择分块／在线softmax或其他等价结构。考虑低精度中间概率和归约对接近零结果的相对误差影响，优先以足够精度累加并说明选择；不要为了速度直接假定低精度中间态满足严格精度。

只修改candidate.py和diagnosis.json。开始写实现前，填写diagnosis.json（当前源码SHA为e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855），注明依据为官方合同及数学语义，记录假设、实现结构、预期和验证方法，并调用工具桥decision --record diagnosis.json。evidence的path可相对run，如workspace/task-description.md；category为knowledge、role为background。不要虚构已有profiling或模拟器。

本次只生成起点并交接。设备评测在宿主审查后执行，因此不要调用preflight或探测环境。生成完成后以短诊断说明数值、布局和UB策略即可。后续若收到编译或精度失败，再围绕失败修改。不要将官方golden复制为候选设备实现。
