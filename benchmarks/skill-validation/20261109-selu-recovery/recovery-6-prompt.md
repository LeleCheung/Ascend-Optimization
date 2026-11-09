继续遵循skills/hgkda/SKILL.md，读取当前candidate.py后直接修改。只在workspace工作，不探索环境，不读取历史最佳源码。最新正式反馈见next-performance-feedback.json（0.312005×，完整36/36及双方42/42通过）；raw等待版80bfefff的正式测量正在完成，不能编造它的成绩。

宿主补查了实际后端源码，纠正上一条“禁止NoWait”的过严指导：triton/backends/ascend/backend_register.py中torch_npu的get_current_stream本身优先 `_npu_getCurrentRawStreamNoWait(device)`，不存在才用 `_npu_getCurrentRawStream(device)`。生成的启动器默认TRITON_ENABLE_TASKQUEUE=true，通过at_npu::native::OpCommand().SetCustomHandler(launch_call).Run()异步发射编译后的Triton kernel。因此NoWait与后端任务队列相配合，不等于缓存旧stream或调用PyTorch目标算子。

请完成两项实际优化：
1. 模块初始化缓存“查询当前stream的函数对象”，优先torch_npu._C._npu_getCurrentRawStreamNoWait，不存在则回退_npu_getCurrentRawStream。每次调用这个函数获得当前stream；禁止缓存返回的句柄/Tensor/结果。保留当前设备/对齐/连续性/调试条件和普通Triton回退。不要改现有准确ABI。
2. 扩大大输入分块。可复用的历史方法参数：FP16最大16384，BF16/FP32最大8192；小/中输入以约8个program为参考，按元素数选择适合的2次幂分块，上限取dtype上限。超小输入仍有mask/正确处理；不要为了省mask越界。这个是方法参数提示，并未提供历史最佳源码。

全部kernel的数学公式和非连续寻址保持不变，只改宿主分块选择及启动元数据路径。仍仅run公开，其他辅助函数单下划线。先保存当前candidate为workspace/stream-wait-parent.py（可用Read+Write，不执行cp等探索命令），再Edit生成candidate.py，更新diagnosis.json记录后端来源和宿主纠正指导。

编辑完成立即工具桥preflight --source 当前candidate.py绝对路径，timeout=600000。前一个正式任务可能持卡锁，等待是正常行为，不要重发。42/42后返回交给宿主路径审查、offset/stream检查及一次完整KGS。不要在未实测前声称达标。
