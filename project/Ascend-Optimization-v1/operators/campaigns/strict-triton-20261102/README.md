# 严格Triton 10＋20重新核验

最新结果为16/30达标：1015原10项6项，1029新增20项10项。9项性能未过线，2项原语义或补测失败，3项因CANN参考不合格／缺失单列。

- [复核简报](复核结果简报.md)：结论、来源、限制与复现入口。
- [10＋20总表](严格Triton30项总表.md)：导师顺序、源码链接、两种参考、后端、完整正确性和补测。
- `delivery.json`、`delivery-audit.json`：结果及原合同、原请求和逐case身份核验。
- `historical-isolation.json`：19项含禁止路径的历史实现隔离记录，原件保留。
- `reference-qualification-audit.json`：25项合格CANN参考的源码、请求、补测重核。
- `compiled-verification.json`：30项实际Triton TTIR与设备二进制核验。
- `strict-implementation-context.json`：28项语义通过的Triton实现上下文与25项CANN参考分开输出；数值通过不代表性能达标。
- `candidates/`：按算子保存冻结源码、运行时检查、完整KGS、补测、trace与实际编译产物。
- `revisions/`：原纯Python QKV、空输入trace和分配误报等旧状态，保留原因。
- `controller-logs/`：控制器与各步骤日志；失败尝试保留。

候选所有设备计算与搬运严格Triton；Python／C++只作宿主胶水。QKV经审查保留C++分配和参数打包，设备核仍是Triton。没有将旧Ascend C／CANN／DMA混合成绩计入本次达标数。

固定FlagGems `d6a8eec473517a3d68157b208eb9c057eb1d4c50`、FlagTree `0.6.0+ascend.gitc286cba6`、CANN 9.0.0。KGS 6.5.0完整调用计时，原输入、容差与参数保持不变。一次新复测见[工具说明](../../../tools/profiling/hgkda/strict_triton/README.md)。
