# uniform：若Philox高位乘法降低低效，可测试等价32位拆分

先保留真实生成器状态和完整seed/offset，应用_lean_seed_offset思路，再检查本机Philox的实际IR/热点。一个可区分的机制假设是：umulhi或64位中间乘法被低效降低。只有实际IR或探针支持时才沿此方向，不能预先断言它是瓶颈。

若命中该机制，可以用16位半字分解计算uint32乘法的高32位，全程只用uint32乘法、加法和移位，保持Philox算法及轮数，避免通过替换成弱哈希来换性能：

```python
x0 = x & 0xffff
x1 = x >> 16
c0 = C & 0xffff  # constexpr
c1 = C >> 16
w0 = x0 * c0
t = x1 * c0 + (w0 >> 16)
w1 = x0 * c1 + (t & 0xffff)
hi = x1 * c1 + (t >> 16) + (w1 >> 16)
lo = x * C
```

所有x及中间量均为tl.uint32，右移必须无符号；C为Philox轮常数0xd2511f53或0xcd9e8d57。CPU固定边界和100000个随机uint32输入、两种常数的200016次检查均与(x*C)>>32一致，证据在research/uniform/philox-mulhi-cpu/。这是有限CPU身份检查，不是NPU加速证明，也不是替代完整算法验证。原生umulhi若已经高效矢量化，该拆分可能反而更慢，不要盲目采用。

若试验，额外核对Philox4x32完整输出与原版一致、seed/offset高位、状态推进与恢复重放，再执行18项语义和一次完整KGS。只有总成本下降才保留。不要为了新seed重编译，不改固定FlagTree，不省略随机轮次；记录实际IR变化及完整耗时。此笔记是第一批结束后Codex的等价表达式备选，尚未在NPU验证。
