加载skills/hgkda/SKILL.md。只在当前workspace工作。直接Read launch-v1.py，然后用Edit/Write保存为candidate.py，进行下面的确定性修复。不要探索环境，不要运行任何ABI猜测或逐个尝试参数槽位。
此前宿主提示的启动器示例来自只有BLOCK一个constexpr的历史kernel，但你的kernel有BLOCK和F16MATH两个constexpr，所以示例少了一个参数。这是宿主提供示例的错误；后端没有变化。当前后端make_launcher源码已核对：format="iiiKKOOOO" + ''.join(format_of(ty) for ty in signature.values()) + debug_abi.parse_format；constexpr也保留一个O槽位。常规9个前缀后，必须按原函数参数顺序传所有显式参数/constexpr占位，不可在packed后增加前缀。
因此只有两种准确的调用：
nomask: launcher.launch(grid,1,1,current_stream,function,packed_metadata,None,None,None,current_pointer,None,None)
masked: launcher.launch(grid,1,1,current_stream,function,packed_metadata,None,None,None,current_pointer,n,None,None)
两个尾None依次占BLOCK、F16MATH。不是隐藏参数。禁止运行时猜测slot数量，禁止捕获TypeError循环重发，禁止catch执行错误后重新执行原地计算。
launch-v1.py是你上次生成的第一次预检版本，kernel计算已正确，宿主代码缓存启动器、每次读取当前指针/stream也合理。只修_do_launch的两个调用并改正ABI注释，其他逻辑先保留。run(A)保持直接调用。注释中文，所有辅助函数单下划线，仅run公开。全部设备计算Triton，不引入CANN/ATen。
将diagnosis补充abi_correction，明确是按kernel显式constexpr保留占位。然后调用下面工具桥preflight --source 当前candidate.py绝对路径，timeout=600000。不加--operator，不执行其他Python脚本。若有真实错误按反馈修复。预检42/42通过后返回候选SHA与结果，宿主继续审查/正式性能评测。无需重新阅读上下文大JSON。

工具桥命令：/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python /data/hanle/ascend-optimization/runs/.queued/selu-skill-recovery-20261109-v2/package/tools/skill_validation/agent_tools.py --config /data/hanle/ascend-optimization/runs/selu-skill-recovery-20261109-v2/config.json