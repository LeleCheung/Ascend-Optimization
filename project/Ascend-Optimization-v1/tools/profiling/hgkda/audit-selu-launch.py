#!/usr/bin/env python3
"""在独立卡锁下检查SELU候选的缓存查找和真实启动路径；不参与性能计分。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def audit(source, expected_sha256, output):
    if output.exists():
        raise ValueError('输出已存在，保留原始诊断并使用新的路径')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError('候选源码SHA不同')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '7':
        raise ValueError('仅允许已获准的物理卡7；调用者必须持本项目卡锁')
    import torch
    import torch_npu  # noqa: F401
    from triton import knobs
    from triton.runtime import driver, jit

    assert torch.npu.device_count() == 1
    spec = importlib.util.spec_from_file_location('selu_launch_candidate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = {'source_sha256': digest,
            'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'jit_sha256': hashlib.sha256(Path(jit.__file__).read_bytes()).hexdigest(),
            'physical_device': 7, 'logical_device': 'npu:0',
            'scope': '单候选启动路径诊断，不修改原始源码，不作为KGS正式正确性或性能复验',
            'status': 'failed'}
    try:
        kernel = module.selu_kernel_
        block, count = 1024, 1024
        generator = torch.Generator().manual_seed(20261021)
        original = torch.randn((count,), generator=generator).to('npu:0', dtype=torch.float16)
        value = original.clone()
        kernel[(1,)](value, count, BLOCK_SIZE=block)
        torch.npu.synchronize()
        device = driver.active.get_current_device()
        kernel_cache, key_cache, _target, _backend, binder = kernel.device_caches[device]
        args = (value, count)
        raw_bound, raw_specialization, raw_options = binder(*args, BLOCK_SIZE=block)
        raw_key = jit.compute_cache_key(key_cache, raw_specialization, raw_options)
        stock_kwargs = {'BLOCK_SIZE': block, 'debug': kernel.debug or knobs.runtime.debug}
        jit._flagprism.apply_compile_options(stock_kwargs)
        stock_bound, stock_specialization, stock_options = binder(*args, **stock_kwargs)
        stock_key = jit.compute_cache_key(key_cache, stock_specialization, stock_options)
        data.update(raw_options=raw_options, stock_options=stock_options,
                    raw_key_hits_compiled_cache=raw_key in kernel_cache,
                    stock_key_hits_compiled_cache=stock_key in kernel_cache,
                    same_specialization=raw_specialization == stock_specialization,
                    same_bound_argument_names=list(raw_bound) == list(stock_bound),
                    same_cache_key=raw_key == stock_key,
                    compiled_cache_entries=len(kernel_cache))
        value.copy_(original)
        expected = torch.selu(original)
        executed = module._fast_launch(kernel, args, block, 1)
        torch.npu.synchronize()
        data['original_fast_launch_executed'] = bool(executed)
        data['original_slow_latch'] = list(module._SLOW)
        torch.testing.assert_close(value, expected if executed else original,
                                   atol=.002, rtol=.002)
        compiled = kernel_cache.get(stock_key)
        if compiled is None:
            raise RuntimeError('按实际JIT选项查找仍未命中编译缓存')
        if hasattr(compiled, 'result'):
            compiled = compiled.result()
        value.copy_(original)
        stream = driver.active.get_current_stream(device)
        metadata = compiled.launch_metadata((1, 1, 1), stream, *stock_bound.values())
        compiled.run(1, 1, 1, stream, compiled.function, compiled.packed_metadata,
                     metadata, knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
                     *stock_bound.values())
        torch.npu.synchronize()
        torch.testing.assert_close(value, expected, atol=.002, rtol=.002)
        data.update(status='passed', stock_key_direct_launch_correct=True)
    except Exception as error:
        data['error'] = type(error).__name__ + ': ' + str(error)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(data, ensure_ascii=False, indent=2)+'\n').encode())
    print(json.dumps(data, ensure_ascii=False))
    return int(data['status'] != 'passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(audit(args.source, args.expected_sha256, args.output))
