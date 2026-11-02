"""在CPU逐句解释小矩阵Triton源码，检查LU/求解索引；不使用NPU或计时。"""
import argparse
import ast
from collections import namedtuple
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--leaf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import numpy as np
    import torch

    class Tensor(np.ndarray):
        __array_priority__ = 1000

        def to(self, dtype, bitcast=False):
            array = np.asarray(self)
            return wrap(array.view(dtype) if bitcast else array.astype(dtype))

    def wrap(value, dtype=None):
        return np.asarray(value, dtype=dtype).view(Tensor)

    class Pointer:
        def __init__(self, base, indices=0):
            self.base, self.indices = base, np.asarray(indices)

        def __add__(self, value):
            return Pointer(self.base, self.indices + np.asarray(value))

    def load(pointer, mask=True, other=0.):
        indices, valid = np.broadcast_arrays(pointer.indices, np.asarray(mask, dtype=bool))
        value = np.full(indices.shape, other, dtype=pointer.base.dtype)
        value[valid] = pointer.base[indices[valid]]
        return wrap(value)

    def store(pointer, value, mask=True):
        indices, value, valid = np.broadcast_arrays(pointer.indices, np.asarray(value),
                                                     np.asarray(mask, dtype=bool))
        pointer.base[indices[valid]] = value[valid]

    def dtype_of(value):
        if isinstance(value, (np.ndarray, np.generic)):
            return value.dtype
        return np.dtype(np.float32 if isinstance(value, float) else np.int32)

    def where(condition, a, b):
        return wrap(np.where(condition, a, b), np.result_type(dtype_of(a), dtype_of(b)))

    current = [0]
    tl = types.SimpleNamespace(
        constexpr=object, float32=np.float32, int32=np.int32, uint32=np.uint32, int64=np.int64,
        program_id=lambda axis: current[0], arange=lambda a, b: wrap(np.arange(a, b, dtype=np.int32)),
        full=lambda shape, value, dtype: wrap(np.full(shape, value, dtype=dtype)),
        range=lambda *values: range(*(int(v) for v in values)),
        static_range=lambda *values: range(*(int(v) for v in values)),
        load=load, store=store, where=where,
        sum=lambda a, axis: wrap(np.sum(a, axis=axis, dtype=a.dtype)),
        max=lambda a, axis: wrap(np.max(a, axis=axis)),
        min=lambda a, axis: wrap(np.min(a, axis=axis)),
        abs=lambda a: wrap(np.abs(a)), div_rn=lambda a, b: wrap(a / b),
    )
    # 只执行原源码中的函数体；去掉编译装饰器和导入，不改数学/索引表达式。
    tree = ast.parse(args.source.read_text(encoding='utf8'))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in tree.body:
        node.decorator_list = []
    namespace = dict(tl=tl, torch=torch, namedtuple=namedtuple)
    exec(compile(tree, str(args.source), 'exec'), namespace)
    kernel = namespace['_leftlooking']
    spec = importlib.util.spec_from_file_location('cpu_inverse_oracle', args.leaf / 'oracle.py')
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    rows = []
    for line in (args.leaf / 'correctness.jsonl').read_text(encoding='utf8').splitlines():
        context = json.loads(line)
        inputs = oracle.gen_inputs(context, 'cpu')
        a = inputs['A']
        n = a.shape[-1]
        count = a.numel() // (n * n)
        inverse = np.empty(a.numel(), dtype=np.float32)
        info = np.empty(count, dtype=np.int32)
        stages = []

        def trace(frame, event, arg):
            if event == 'return' and frame.f_code.co_name == '_leftlooking':
                stages.append({name: np.asarray(frame.f_locals[name]).tolist()
                               for name in ['work', 'rhs', 'pivots']})
            return trace

        try:
            for index in range(count):
                current[0] = index
                sys.settrace(trace)
                try:
                    kernel(Pointer(a.numpy().reshape(-1)), Pointer(inverse), Pointer(info), n,
                           max(16, 1 << (n - 1).bit_length()), tuple(a.shape[:-2]), tuple(a.stride()), True)
                finally:
                    sys.settrace(None)
            actual = torch.from_numpy(inverse.reshape(tuple(a.shape)))
            expected = oracle.run(**inputs)
            row = dict(case=context['name'], input=a.tolist(), actual=actual.tolist(),
                       reference=expected.inverse.tolist(), info=info.tolist(),
                       reference_info=expected.info.tolist(), stages=stages)
            try:
                tolerance = context['tolerance']
                torch.testing.assert_close(actual, expected.inverse, atol=tolerance['atol'],
                                           rtol=tolerance['rtol'], equal_nan=True)
                torch.testing.assert_close(torch.from_numpy(info.reshape(tuple(a.shape[:-2]))), expected.info)
                row['passed'] = True
            except AssertionError as error:
                row.update(passed=False, error=str(error)[:2000])
        except Exception as error:
            row = dict(case=context['name'], passed=False, error=type(error).__name__ + ': ' + str(error))
        rows.append(row)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    value = dict(status='cpu_literal_kernel_model', source_sha256=sha(args.source),
                 checker_sha256=sha(__file__), rows=rows, device_executed=False,
                 scope='原函数体在CPU用NumPy指针/FP32算术模型执行；仅诊断索引与更新顺序，不是厂商解释器或NPU成绩。')
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps([{k: row[k] for k in ['case', 'passed', 'error'] if k in row} for row in rows]))


if __name__ == '__main__':
    main()
