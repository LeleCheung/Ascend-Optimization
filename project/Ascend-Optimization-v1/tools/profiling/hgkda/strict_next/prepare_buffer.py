"""生成保留数学表达的单缓冲大块编译探测；不提交设备评测。"""
import ast
import argparse
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
BATCH = PROJECT / 'operators/campaigns/strict-next-20261103'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retain-fp16', action='store_true', help='继承FP16原分块与编译选项，仅调整BF16和FP32')
    args = parser.parse_args()
    parent = BATCH / 'starting-points/expm1_.py'
    text = parent.read_text(encoding='utf8')
    changes = [
        ('torch.float32: (4096, 2)', 'torch.float32: (8192, 1)'),
        ('ck = kernel.warmup(A, n, tiny, BLOCK=block, grid=(1,))',
         ('options = {} if A.dtype == torch.float16 else dict(num_stages=1, enable_ubuf_saving=True)\n'
          '            ck = kernel.warmup(A, n, tiny, BLOCK=block, grid=(1,), **options)') if args.retain_fp16 else
         'ck = kernel.warmup(A, n, tiny, BLOCK=block, grid=(1,), num_stages=1, enable_ubuf_saving=True)'),
    ]
    if not args.retain_fp16:
        changes.insert(0, ('torch.float16: (8192, 2)', 'torch.float16: (16384, 1)'))
    for before, after in changes:
        assert text.count(before) == 1
        text = text.replace(before, after, 1)
    ast.parse(text)
    out = BATCH / ('compile-probes/single-buffer-layout-v2' if args.retain_fp16 else 'compile-probes/single-buffer-layout-v1')
    assert not out.exists(), '已有编译探测，不能覆盖'
    out.mkdir(parents=True)
    source = out / 'main.py'
    source.write_text(text, encoding='utf8', newline='\n')
    config = []
    for dtype, tiny, block in [('fp16', .125, 16384), ('bf16', .125, 8192), ('fp32', .0078125, 8192)]:
        stream = '_expm1_streaming1'
        options = dict(num_warps=32, num_stages=1, enable_ubuf_saving=True)
        if args.retain_fp16 and dtype == 'fp16':
            stream, block, options = '_expm1_streaming2', 8192, dict(num_warps=32)
        for symbol, tile in [('_expm1_flat', 4096), (stream, block)]:
            config.append(dict(dtype=dtype, tiny=tiny, symbol=symbol, block=tile,
                               options=options))
    (out / 'compile-configuration.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf8')
    spec = dict(source_sha256=digest(source), parent_source_sha256=digest(parent),
                preparer_sha256=digest(Path(__file__)), changes=changes,
                attribution='Codex基于相同GM总量、较低搬运粒度及Vector额外开销提出缓冲组织假设',
                hypothesis='保持小值数学、阈值与全部布局；单阶段及UB复用可能容纳较大连续块，减少搬运碎片及循环控制，收益待真机验证',
                limitations='未关闭必要同步，不改自定义编译流水线；能编译不代表更快；原注释内历史测量不是本稿成绩',
                retained_fp16=args.retain_fp16,
                previous_failure='v1的FP16单块16384在单阶段和UB优化下仍超出192KiB；本稿不再尝试该路径' if args.retain_fp16 else None,
                device_executed=False, performance_measured=False, qualified=False)
    (out / 'spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(status='compile_probe_prepared', source_sha256=digest(source))))


if __name__ == '__main__':
    main()
