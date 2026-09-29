# 优化器的多语言只读代码参考

SimpleOpt、KernelGen 与 CatalogOptimize 使用 reference_code_path 提供只读实现参考，可选 reference_code_prompt_path 补充来源和迁移限制。来源可以是 Triton、CUDA、Ascend C、Torch 等文本；KG 只读取，不 import、编译或执行，也不要求源文件后缀为 .py。Python Batch 按 <definition>.<extension> 绑定显式文件，不扫描目录。

CLI 与 Python launcher 使用 --reference-code-path 和 --reference-code-prompt-path，YAML 使用对应下划线字段。原 --reference-triton-path/--reference-triton 及旧 JSON 字段只在输入解析边界兼容，内部和新序列化输出统一使用 reference_code_*。不维护两套加载或提示生成逻辑。

来源代码可能错误、不可移植或慢于基线，只是实现参考，不改变生成目标语言，不成为 seed、oracle 或 timing baseline。Coder 必须保持 Definition、workload、validator 和目标 KGS 契约，来源说明不能覆盖这些要求。文本大小、编码、空文件和只读校验沿用原逻辑。

seed 仍是独立输入；本改动不更改其评测语义，也不代表统一 CatalogOptimize CLI 入口、seed_triton 参数收敛或新的代码冻结恢复方案已完成。旧 campaign 不迁移 workspace 或重写历史请求。
