加载skills/hgkda/SKILL.md和references/launch-and-feedback.md，读取starter.py及launch-diagnostic-brief.json，然后直接用Write/Edit生成candidate.py。上次调用耗尽轮次仅因为探索了不存在/不允许的环境路径，没有产生候选，本次不要再探索环境或执行probe。已提供当前编译器的ABI，用预检验证即可。
这是宿主Python3.12控制模型，实际设备Python3.11在容器；不得在宿主import triton或探查venv，所有设备实验只通过下面工具桥preflight。
只在当前workspace读写。不要读取仓库、其他实验或历史最佳源码，不使用其他agent。不要重读巨大IR、knowledge.json和context全文。精简证据已在launch-diagnostic-brief.json。
任务：从starter.py保留全部kernel计算、参数和非连续分支，只改连续分支宿主启动缓存。run(A)直接调用_selu_inplace(A)，不再每次解析*args/**kwargs。第一次warmup仅编译，不能重复执行原地计算。
本机支持：compiled=kernel.warmup(A,BLOCK=block,F16MATH=f16_math,grid=(1,))；launcher=compiled.run，为NPULauncher。nomask调用launcher.launch(grid,1,1,current_stream,compiled.function,compiled.packed_metadata,None,None,None,A.data_ptr(),None)。masked的warmup传A,n，launch指针后传n，再None。最后None是后端隐藏参数，不能省略。
缓存仅含编译句柄、网格、元数据、参数模板，不含Tensor/旧stream/结果。按device/dtype/shape/alignment等分开，每次取当前A指针和torch_npu.npu.current_stream(A.device).npu_stream。对齐连续路径缓存，未对齐及非连续走普通Triton。NPULauncher compile_only、enable_msprof_register_tensor、metadata.debug_enabled为真时回退普通Triton，避免绕开调试。设备上下文仅首次编译或切设备时进入；缓存命中仍确保当前设备对应输入。
所有设备计算、复制及回退严格Triton，无CANN/ATen。只允许torch/torch_npu/triton/math/functools/typing导入。Python辅助函数全单下划线，唯一公开入口run(A)。注释中文。写diagnosis.json记录证据、机制、修改、预期和使用历史方法。
立即编写candidate.py和diagnosis.json，然后执行工具桥preflight，命令后只增加preflight --source <当前candidate.py绝对路径>，timeout=600000。不加--operator，不加context选项，不运行其他Python。失败根据输出修改，最多6个新SHA。42/42通过后返回，宿主审查和正式评测后将继续性能反馈；不要声明goal完成。

工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/.queued/selu-skill-recovery-20261109-v2/package/tools/skill_validation/agent_tools.py --config /data/hanle/ascend-optimization/runs/selu-skill-recovery-20261109-v2/config.json