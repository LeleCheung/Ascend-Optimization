"""在完整块Triton提案上复用已核验的元数据/启动桥，不新增设备实现。"""
import copy
import hashlib
import json
from pathlib import Path


BATCH = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')


def main():
    parent = BATCH / 'proposals/expm1_/codex-fulltiles-expm1-v2/main.py'
    assert sha(parent) == '227536027fa11bb81c575aab1789da0815ce4688e06cac4d4bcd10179dcc2393'
    donor = BATCH / 'proposals/log2_/codex-vectorcall-host-v3'
    donor_text = (donor / 'main.py').read_text(encoding='utf8')
    start = donor_text.index('# 人工核验的宿主桥：')
    end = donor_text.index('_python_triton_run = run', start) + len('_python_triton_run = run')
    loader = donor_text[start:end]
    text = parent.read_text(encoding='utf8') + '\n\n' + loader + '''


def _bridge_fallback(A):
    # 冷调用只执行一次原Triton入口；登记计划不运行设备代码。
    output = _python_triton_run(A)
    n = A.numel()
    device = A.get_device()
    if n and A.is_contiguous() and not (A.data_ptr() & 15):
        entry = _ENTRIES.get((A.dtype, n, device))
        # 父入口在调试/注册模式下缓存launcher包装对象；这些模式继续走原入口。
        if entry is not None and getattr(entry[0], 'metadata', None) is None:
            arguments = (entry[3], 1, 1, None, entry[1], entry[2],
                         None, None, None, None, n, entry[4])
            _BRIDGE.register_plan(A, entry[0], arguments,
                                  lambda: _current_stream(device))
    return output


_BRIDGE = _bridge_module.InplaceLaunchCache(_bridge_fallback)


def run(A):
    return _BRIDGE.run(A)
'''
    folder = BATCH / 'proposals/expm1_/codex-vectorcall-expm1-v3'
    assert not folder.exists(), '已有提案原件，不覆盖'
    folder.mkdir(parents=True)
    source = folder / 'main.py'
    source.write_text(text, encoding='utf8', newline='\n')
    attribution = 'Codex复用已审查的C++元数据/Vectorcall启动桥，减少expm1小输入的Python路径；设备公式及全部回退逐字继承完整块Triton父稿。仅父稿数值通过后执行。'
    save(folder / 'spec.json', dict(operator='expm1_', source_sha256=sha(source),
        parent_source_sha256=sha(parent), host_glue=True, requires_qualified_parent=True,
        defer_if_target_already_met=True, attribution=attribution))
    donor_review = json.loads((donor / 'review.json').read_text(encoding='utf8'))
    bridges = copy.deepcopy(donor_review['reviewed_host_bridges'])
    for bridge in bridges:
        bridge['candidate_sha256'] = sha(source)
        bridge['functions'] = ['Key', 'KeyHash', 'Plan', 'InplaceLaunchCache', 'register_plan',
                               'run', 'stats', 'checked', 'key', 'eligible', 'slow', 'PYBIND11_MODULE']
    save(folder / 'review.json', dict(operator='expm1_', source_sha256=sha(source),
        status='reviewed_strict_triton', all_branches_reviewed=True, dependencies_reviewed=True,
        reviewer='Codex逐函数复查完整Python父稿及host_bridge.cpp全部函数',
        reviewed_host_bridges=bridges, attribution=attribution, review_notes=[
            '设备Triton函数逐字继承v2，包含完整块、尾块及非连续分支；新C++部分不包含设备运算、搬运、CANN或Ascend C。',
            'InplaceLaunchCache仅读取numel/dtype/device/连续性/当前地址，查不可变启动计划并Vectorcall到既有Triton生成启动器。',
            '每次动态填入当前stream与本次Tensor地址，注册参数中相应两槽为None；计划不保留Tensor、输出或stream值。',
            '冷调用只执行一次父入口，之后仅登记计划；提交错误直接传播，绝不重做原地计算。',
            '调试launcher包装对象不登记快路径；空、非连续、非对齐输入继续走父Triton路径。',
            '固定参数来自父稿实际编译对象，缓存键包含dtype、numel与device；N与BLOCK随该二进制计划冻结。',
            '模块缓存核验实际加载路径，并绑定二进制、C++源文件、构建记录及Torch版本/ABI身份。',
            'v2尚未NPU数值验证，v3需等父稿正式/补测通过；不把源码继承或宿主审查写成NPU性能结果。']))
    print(json.dumps(dict(status='prepared', source_sha256=sha(source), parent_source_sha256=sha(parent))))


if __name__ == '__main__':
    main()
