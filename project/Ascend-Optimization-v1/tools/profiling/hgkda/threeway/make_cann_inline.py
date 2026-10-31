"""归一化CANN参考v3：query/execute同一宿主线程，避免跨线程执行器/资源持有。"""
from pathlib import Path


here=Path(__file__).resolve().parent
source=(here/'cann_cpp.cpp').read_text(encoding='utf8')
old='    auto out=at::empty(input.sizes(),input.options());'
assert source.count(old)==1
source=source.replace(old,'''    auto raw_stream=c10_npu::getCurrentNPUStream(input.device().index()).stream();
'''+old)
old='    dispatch("aclnnGroupNorm",tensors,executor,bytes,input,aclnnGroupNorm);'
assert source.count(old)==1
source=source.replace(old,'''    // 与有效ctypes参考保持相同线程和宿主队列提交顺序，移除Python事件及转换成本。
    // 资源来自当前stream分配器，execute完成提交后按同stream顺序复用；不等待设备。
    at::Tensor workspace;
    if (bytes) workspace=at::empty({static_cast<int64_t>(bytes)},input.options().dtype(at::kByte));
    code=aclnnGroupNorm(workspace.defined()?workspace.data_ptr():nullptr,bytes,executor,raw_stream);
    TORCH_CHECK(code==0,"aclnnGroupNorm execute failed: ",code);''')
target=here/'cann_cpp_group_norm_v3.cpp'
if target.exists() and target.read_text(encoding='utf8')!=source:raise FileExistsError(target)
target.write_text(source,encoding='utf8')
print(target)
