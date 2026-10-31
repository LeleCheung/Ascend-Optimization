"""归一化参考v2：分配下一份输出前排空同stream宿主队列，不等待设备。"""
from pathlib import Path


here = Path(__file__).resolve().parent
source = (here/'cann_cpp.cpp').read_text(encoding='utf8')
old = '    auto out=at::empty(input.sizes(),input.options());'
assert source.count(old) == 1
source = source.replace(old, '''    // RunOpApiV2闭包持有新输出及工作区；连续暖机时，先提交前次宿主任务，
    // 避免大量尚未执行的闭包同时持有整张量。不调用设备synchronize。
    c10_npu::getCurrentNPUStream(input.device().index()).stream();
'''+old)
target = here/'cann_cpp_group_norm_v2.cpp'
if target.exists() and target.read_text(encoding='utf8') != source:
    raise FileExistsError(target)
target.write_text(source,encoding='utf8')
print(target)
