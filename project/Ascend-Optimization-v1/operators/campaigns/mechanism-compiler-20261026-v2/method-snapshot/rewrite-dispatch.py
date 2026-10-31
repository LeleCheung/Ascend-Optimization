"""将已证实的调用分派假设编译为通用缓存变换；保留原数学内核和未知路径。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

# 该模板不包含算子公式，不缓存输入Tensor、地址或stream。首调用只执行原实现一次。
TEMPLATE=r'''

# HG-KDA 通用调用变换；只支持单输入原地、一个JIT提交的已观测路径。
import threading as _hg_threading
import torch as _hg_torch
import torch_npu as _hg_torch_npu
from triton.runtime.jit import JITFunction as _hg_JITFunction
from triton.runtime import driver as _hg_driver
from triton.runtime import jit as _hg_jit_module
from triton import knobs as _hg_knobs

_hg_original_run = run
_hg_entries = {}
_hg_lock = _hg_threading.RLock()


def _hg_compile_plan(value):
    captures = []
    kernels = []
    for symbol in tuple(globals().values()):
        if isinstance(symbol, _hg_JITFunction) and symbol.fn.__module__ == __name__:
            kernels.append(symbol)
        elif hasattr(symbol, 'prepare_args') and hasattr(symbol, 'materialize'):
            try:
                ndim, _, _ = symbol.prepare_args(value, out0=value)
                kernels.append(symbol.materialize(ndim).jit_function)
            except (TypeError, ValueError, AssertionError, AttributeError):
                pass
    kernels = list({id(kernel): kernel for kernel in kernels}.values())
    saved = []
    def wrap(jit, original):
        def wrapped(*args, grid, warmup, **kwargs):
            compiled = original(*args, grid=grid, warmup=warmup, **kwargs)
            if not warmup:
                options = dict(kwargs)
                options['debug'] = options.get('debug', jit.debug) or _hg_knobs.runtime.debug
                _hg_jit_module._flagprism.apply_compile_options(options)
                device = _hg_driver.active.get_current_device()
                binder = jit.device_caches[device][4]
                bound, _, _ = binder(*args, **options)
                actual_grid = grid(bound) if callable(grid) else grid
                if hasattr(compiled, 'result'):
                    compiled = compiled.result()
                captures.append((compiled, tuple(actual_grid), tuple(bound.values())))
            return compiled
        return wrapped
    # 只代理本候选自己的JIT实例；不改torch/triton模块、类或全局运行时。
    for kernel in kernels:
        had_local = 'run' in kernel.__dict__
        original = kernel.run
        saved.append((kernel, had_local, original))
        kernel.run = wrap(kernel, original)
    try:
        result = _hg_original_run(value)
    finally:
        for kernel, had_local, original in saved:
            if had_local:
                kernel.run = original
            else:
                del kernel.run
    # 多个提交可能来自autotune或临时量，不能猜测哪些可以直接重放。
    if result is not value or len(captures) != 1:
        return result, None
    compiled, grid, bound = captures[0]
    template = []
    for argument in bound:
        if isinstance(argument, _hg_torch.Tensor) or (hasattr(argument, 'data_ptr') and hasattr(argument, 'dtype') and hasattr(argument, 'device')):
            if argument.device != value.device or argument.dtype != value.dtype or argument.data_ptr() != value.data_ptr():
                return result, None
            template.append(('input', None))
        elif isinstance(argument, int) and not isinstance(argument, bool) and argument == value.data_ptr():
            template.append(('pointer', None))
        elif isinstance(argument, (int, float, bool, str, type(None))):
            template.append(('constant', argument))
        else:
            return result, None
    return result, (compiled, grid, tuple(template))


def run(A):
    if not A.is_contiguous() or A.numel() == 0:
        return _hg_original_run(A)
    # 形状、stride、dtype、设备与地址对齐属于特化身份；stream每次重取。
    key = (str(A.device), A.dtype, tuple(A.shape), tuple(A.stride()), A.data_ptr() % 4096)
    with _hg_lock:
        entry = _hg_entries.get(key)
        if entry is None:
            # 首次编译路径已经执行，必须立刻返回，禁止再提交一次。
            with _hg_torch.npu.device(A.device):
                result, entry = _hg_compile_plan(A)
            if entry is not None:
                _hg_entries[key] = entry
            return result
        compiled, grid, template = entry
        bound = tuple(A if kind == 'input' else A.data_ptr() if kind == 'pointer' else val
                      for kind, val in template)
        with _hg_torch.npu.device(A.device):
            device = _hg_driver.active.get_current_device()
            stream = _hg_driver.active.get_current_stream(device)
            metadata = compiled.launch_metadata(grid, stream, *bound)
            padded = grid + (1,) * (3-len(grid))
            compiled.run(*padded, stream, compiled.function, compiled.packed_metadata, metadata,
                         _hg_knobs.runtime.launch_enter_hook, _hg_knobs.runtime.launch_exit_hook, *bound)
        return A

__OPERATOR__ = run
'''


def rewrite(source,operator,diagnosis,output):
    if output.exists():raise ValueError('不覆盖已有候选')
    if not re.fullmatch('[a-z][a-z0-9_]*',operator):raise ValueError('算子名非法')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    facts=json.loads(diagnosis.read_text(encoding='utf8'))
    if facts.get('source_sha256')!=digest:raise ValueError('诊断与起点源码不一致')
    supported=[d for d in facts['decisions'] if any(h.get('id')=='dispatch' and h.get('supported') is True
                                                for h in d.get('competing_hypotheses',[]))]
    if not supported:raise ValueError('没有干预证据支持分派变换，拒绝自动改写')
    original=source.read_text(encoding='utf8')
    if '_hg_original_run' in original:raise ValueError('输入已应用该变换')
    transformed=original+'\n'+TEMPLATE.replace('__OPERATOR__',operator)
    ast.parse(transformed)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(transformed.encode())
    provenance=dict(transform='single-input-inplace-cached-dispatch-v1',operator=operator,
        parent_source_sha256=digest,source_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        transformer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnosis_sha256=hashlib.sha256(diagnosis.read_bytes()).hexdigest(),supported_cases=[d['case'] for d in supported],
        invariants=['首调用只执行一次','保留原数学内核和未知布局回退','缓存不含输入Tensor、数据地址或stream',
                    '实际输入/指针与当前stream逐次重新绑定','必须通过独立语义和完整KGS评测'])
    output.with_suffix('.rewrite.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(provenance,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','diagnosis','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--operator',required=True);a=p.parse_args()
    rewrite(a.source,a.operator,a.diagnosis,a.output)
