"""只改最新最佳的小值多项式表达，保留全部布局、分块、启动与回退路径。"""
import argparse
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
BATCH = PROJECT / 'operators/campaigns/strict-next-20261103'
REMOTE = '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--typed', action='store_true')
    args = parser.parse_args()
    source = BATCH / 'starting-points/expm1_.py'
    old = source.read_text(encoding='utf8')
    before = 'return tl.where(tl.abs(x) < TINY, x * (1.0 + 0.5 * x), tl.exp(x) - 1.0)'
    after = 'return tl.where(tl.abs(x) < TINY, x * tl.fma(x, 0.5, 1.0), tl.exp(x) - 1.0)'
    if args.typed:
        after = ('return tl.where(tl.abs(x) < TINY, '
                 'x * tl.fma(x, tl.full((), 0.5, x.dtype), tl.full((), 1.0, x.dtype)), tl.exp(x) - 1.0)')
    assert old.count(before) == 1
    text = old.replace(before, after, 1)
    folder = BATCH / 'proposals/expm1_' / ('codex-typed-fma-v2' if args.typed else 'codex-fma-series-v1')
    assert not folder.exists(), '已有提案，不能覆盖源码身份'
    folder.mkdir(parents=True)
    target = folder / 'main.py'
    target.write_text(text, encoding='utf8', newline='\n')
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    parent = hashlib.sha256(source.read_bytes()).hexdigest()
    record = dict(operator='expm1_', source=REMOTE + target.relative_to(PROJECT).as_posix(), source_sha256=digest,
                  parent_source_sha256=parent, status='reviewed_strict_triton', all_branches_reviewed=True,
                  dependencies_reviewed=True, reviewer='Codex逐路径继承已核验8ec3ef796d4bdbe7，核对唯一函数体差异及依赖',
                  method='x*(1+0.5*x)改为x*tl.fma(x,0.5,1)，保留乘法形式的零符号路径；准确性与收益待设备验证',
                  reviewed_change=dict(before=before, after=after),
                  typed_constants=args.typed,
                  boundaries='只使用Triton数学内置；不调用外部Ascend C/CANN/原生DMA；所有布局与回退继承已审查源码',
                  scope='源码审查不是数值或性能通过；原源码文档字符串的历史测量不作为本稿新成绩')
    (folder / 'review.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    spec = dict(source_sha256=digest, parent_source_sha256=parent, attribution='Codex基于新FP16成对指标和实际IR提出显式乘加表达',
                hypothesis='减少小值多项式中独立向量算术与临时值；后端是否融合及实际收益需验证',
                typed_constants=args.typed,
                previous_compile_failure='v1默认常量导致FMA与结果提升到FP32，流式FP16/BF16产生UB溢出；v2显式同dtype常量' if args.typed else None,
                unchanged='数学阈值、分块、布局、启动、输入、容差和原KGS合同')
    (folder / 'spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(status='prepared', source_sha256=digest)))


if __name__ == '__main__':
    main()
