"""针对BF16指数先舍入的消减误差提出宽精度计算；保留严格Triton所有路径。"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
BATCH = PROJECT / 'operators/campaigns/strict-next-20261103'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')


def main():
    parent = BATCH / 'compile-probes/single-buffer-layout-v2/main.py'
    source = parent.read_text(encoding='utf8')
    before = 'return tl.where(tl.abs(x) < TINY, x * (1.0 + 0.5 * x), tl.exp(x) - 1.0)'
    after = ('if x.dtype == tl.bfloat16:\n'
             '        wide = x.to(tl.float32)\n'
             '        value = tl.where(tl.abs(wide) < TINY, wide * (1.0 + 0.5 * wide), tl.exp(wide) - 1.0)\n'
             '        return value.to(x.dtype)\n'
             '    ' + before)
    assert source.count(before) == 1
    source = source.replace(before, after, 1)
    ast.parse(source)
    folder = BATCH / 'compile-probes/bf16-wide-v3'
    assert not folder.exists(), '已有提案，不覆盖'
    folder.mkdir(parents=True)
    target = folder / 'main.py'
    target.write_text(source, encoding='utf8', newline='\n')
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    config = (parent.parent / 'compile-configuration.json').read_bytes()
    (folder / 'compile-configuration.json').write_bytes(config)
    spec = importlib.util.spec_from_file_location('bf16_wide_history', HERE.parent / 'strict_night/measurements.py')
    history = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(history)
    candidates = []
    for label in ['strict-night-20261102', 'strict-five-20261103', 'strict-next-20261103']:
        root = BATCH.parent / label
        for name in ['candidates/expm1_', 'proposals/expm1_', 'compile-probes']:
            candidates.extend(p for p in (root / name).glob('*/main.py') if p != target)
    normalized = history.computation_identity(target)
    duplicates = [str(p) for p in candidates if history.computation_identity(p) == normalized]
    assert not duplicates
    save(folder / 'history-admission.json', dict(status='new_computation_identity', source_sha256=digest,
         normalized_AST_sha256=normalized, historical_sources_checked=len(candidates), duplicates=[]))
    save(folder / 'review.json', dict(source_sha256=digest, status='reviewed_strict_triton',
         all_branches_reviewed=True, dependencies_reviewed=True, reviewer='Codex逐路径差异审查',
         changes='继承v2分块与缓冲；仅BF16数学在FP32内完成，最终转换回BF16。FP16及FP32数学、所有布局与启动保持父版。',
         boundaries='Triton内部类型转换和算术，未引入外部计算；不缓存数据或stream，设备、地址和长度缓存身份沿用已核验路径。',
         scope='人工准入不代表数值或性能通过；CPU舍入模型必须由真实设备域检查核验。'))
    save(folder / 'spec.json', dict(source_sha256=digest, parent_source_sha256=history.sha(parent),
         attribution='Codex根据BF16逐操作舍入模型、实际TTIR的BF16指数减法和三类型成对指标提出修正',
         hypothesis='BF16先转FP32，指数与减一完成后只舍入一次，避免近零消减误差；宽精度持续计算也可能减少逐操作转换，性能待测。',
         unchanged='FP16及FP32数学、全部小值阈值、布局、指针对齐处理、原地身份、原合同及容差',
         device_executed=False, performance_measured=False))
    print(json.dumps(dict(status='prepared', source_sha256=digest)))


if __name__ == '__main__':
    main()
