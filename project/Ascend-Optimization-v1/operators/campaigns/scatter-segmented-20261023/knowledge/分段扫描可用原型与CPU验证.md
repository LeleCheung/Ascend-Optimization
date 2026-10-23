# 分段扫描原型

Codex已准备`scatter-segmented-prototype.py`，SHA256为`f0ac63a45b09a50860037edec5014f5547fd6bed9806bdd1d49e0603a23becaf`。它是待验证候选，不是最终交付。与原8a67版相比，只改排序后的分段累加及段尾读取，保留片上二分、移位解码、通用非末轴路径。

扫描输入为`(value, head_flag)`，段首标志由相邻排序目标不同产生；组合为`(where(right_head,right_sum,left_sum+right_sum), left_head|right_head)`。段尾直接作为该目标的总和；没有贡献的目标增量为零。这样不会再用其他目标的大前缀做减法。

`scatter-segmented-cpu-model.py`通过显式Hillis–Steele扫描模拟这一组合。`scatter-segmented-cpu.json`记录54/54：三dtype、257/1024/4096三种长度，以及两种随机种子、正负大数、空目标和交替目标六种模式。这只是CPU数学模型，不证明NPU编译、性能或完整正确性。

可以直接用此原型做本机编译探针；若tuple关联扫描不受当前后端支持，保存错误并尝试受支持的分组实现。不得退回全行前缀相减。所有完整原门禁、性能输入数值及动态范围门禁仍须实际通过，再正式测一次完整调用成绩。
