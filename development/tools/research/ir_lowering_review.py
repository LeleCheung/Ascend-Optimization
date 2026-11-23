"""在CPU编译入口核对模型IR的低层产物，单列于真机完整评测。"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(value):
    if isinstance(value, dict):
        if set(value) == {'__tuple__'}:
            return tuple(restore(item) for item in value['__tuple__'])
        return {key: restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore(item) for item in value]
    return value


def target_fields(value):
    if isinstance(value, dict):
        return value
    expression = ast.parse(value, mode='eval').body
    assert isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
    assert expression.func.id == 'GPUTarget' and not expression.args
    fields = {item.arg: ast.literal_eval(item.value) for item in expression.keywords}
    assert set(fields) == {'backend', 'arch', 'warp_size'}
    return fields


def comparison_ir(entry, call, baseline):
    if baseline == 'captured_original':
        path = Path(entry['original'])
        assert sha(path) == entry['original_sha256'], 'Captured original IR identity changed'
        return path
    assert baseline == 'invocation_input'
    return call / 'input-ir' / entry['path']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--baseline', choices=['invocation_input', 'captured_original'],
                        default='invocation_input')
    args = parser.parse_args()
    run = args.run.resolve()
    call = run / 'agent-calls' / args.invocation
    inventory = json.loads((run / 'workspace/ir-inventory.json').read_bytes())
    review = json.loads((call / 'ir-review.json').read_bytes())
    assert review['decision'] == 'approved' and review['source_sha256'] == inventory['source_sha256']
    assert sha(run / 'workspace/frozen.py') == inventory['source_sha256']
    output = call / 'lowering-review'
    output.mkdir(exist_ok=False)
    os.environ['TRITON_CACHE_DIR'] = str(output / 'cache')
    os.environ['TRITON_KERNEL_OVERRIDE'] = '0'
    import torch
    import torch_npu
    import triton
    import triton.compiler.compiler as compiler
    from triton import knobs
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from ir_override_runtime import compiler_identity
    assert not torch.npu.is_initialized(), 'This review uses the CPU compiler entry only'
    torch.set_num_threads(1)
    knobs.cache.dir = str(output / 'cache')
    knobs.compilation.override = False
    native_ir_init = compiler.IRSource.__init__

    def ir_init(instance, path, context, backend):
        # 文件入口与AST入口采用相同的已安装方言注册集合。
        compiler.buffer_ir.load_dialects(context)
        compiler.ascend_ir.load_dialects(context)
        compiler._flagprism.load_dialects(context)
        native_ir_init(instance, path, context, backend)

    compiler.IRSource.__init__ = ir_init
    record = dict(status='compiling', source_sha256=inventory['source_sha256'],
        review_tool_sha256=sha(Path(__file__)), comparison_baseline=args.baseline,
        compiler_identity=compiler_identity(), rows=[], npu_initialized=False,
        scope='IRSource lowering with captured target/options; actual AST injection, correctness and performance are separate')

    def save():
        (call / 'lowering-review.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')

    save()
    for entry in inventory['entries']:
        if entry['path'] not in review['changed_ir']:
            continue
        for name, source in [('original', comparison_ir(entry, call, args.baseline)),
                             ('candidate', call / 'output-ir' / entry['path'])]:
            if name == 'candidate':
                assert sha(source) == review['changed_ir'][entry['path']]
            target = GPUTarget(**target_fields(entry['target']))
            row = dict(variant=name, path=entry['path'], input_ir_path=str(source), input_ir_sha256=sha(source),
                target=target_fields(entry['target']), options=entry['compiler_options'])
            try:
                kernel = compiler.compile(str(source), target=target, options=restore(entry['compiler_options']))
                destination = output / name / entry['ast_sha256']
                destination.mkdir(parents=True)
                artifacts = []
                for key, path in kernel.metadata_group.items():
                    original = Path(path)
                    if not original.is_file():
                        continue
                    copied = destination / original.name
                    copied.write_bytes(original.read_bytes())
                    artifacts.append(dict(kind=original.suffix[1:], path=str(copied.relative_to(call)),
                        sha256=sha(copied), bytes=copied.stat().st_size))
                assert any(item['kind'] == 'npubin' for item in artifacts)
                row.update(status='lowered_to_binary', artifacts=artifacts)
            except Exception as error:
                row.update(status='lowering_failed', error=type(error).__name__ + ': ' + str(error),
                    traceback=traceback.format_exc())
            record['rows'].append(row)
            assert not torch.npu.is_initialized()
            save()
    record['status'] = 'lowering_review_passed' if all(row['status'] == 'lowered_to_binary'
        for row in record['rows']) else 'lowering_review_failed'
    assert record['rows']
    save()
    print(json.dumps(dict(status=record['status'], rows=len(record['rows']), npu_initialized=False)))


if __name__ == '__main__':
    main()
