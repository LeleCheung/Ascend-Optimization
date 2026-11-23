"""Capture actual compiler specializations from a frozen operator without editing it."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def normalized(value):
    from ir_override_runtime import normalized as normalize_compiler_value
    return normalize_compiler_value(value)


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'leaf', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', help='One fixed case, otherwise the complete contract')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['TRITON_CACHE_DIR'] = str(args.output / 'cache')
    os.environ['TRITON_KERNEL_OVERRIDE'] = '0'
    import torch
    import torch_npu
    from torch.utils._python_dispatch import TorchDispatchMode
    from triton import knobs
    from triton.backends.ascend.compiler import NPUOptions
    from triton.runtime.jit import JITFunction
    from ir_override_runtime import compiler_identity
    torch.set_num_threads(1)
    identity = compiler_identity()
    option_names = [field.name for field in dataclasses.fields(NPUOptions)]

    class CandidateOnly(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            allowed = {'aten.empty', 'aten.empty_like', 'aten.empty_strided', 'aten.new_empty', 'aten.alias',
                'aten.detach', 'aten.view', 'aten._unsafe_view', 'aten.as_strided', 'aten.slice',
                'aten.select', 'aten.transpose', 'aten.permute', 'aten.squeeze', 'aten.unsqueeze',
                'aten.resize_', 'aten.resize_as_'}
            if '.'.join(str(function).split('.')[:2]) not in allowed:
                raise RuntimeError('Unexpected target computation outside Triton: ' + str(function))
            return function(*args, **(kwargs or {}))

    candidate = imported('_ir_frozen_candidate', args.source)
    oracle = imported('_ir_contract_inputs', args.leaf / 'oracle.py')
    cases = []
    for name in ['correctness.jsonl', 'timing.jsonl']:
        cases.extend(json.loads(line) for line in (args.leaf / name).read_text().splitlines() if line.strip())
    if args.case:
        cases = [case for case in cases if case['name'] == args.case]
        assert len(cases) == 1, 'Case identity must match exactly'
    active_case = None
    entries = {}
    case_uses = []
    old_listener = knobs.compilation.listener
    old_run = JITFunction.run

    def listener(src, metadata, metadata_group, times, cache_hit):
        if old_listener:
            old_listener(src=src, metadata=metadata, metadata_group=metadata_group,
                times=times, cache_hit=cache_hit)
        cache_hash = metadata['hash']
        if cache_hash in entries:
            return
        ast_hash = src.hash()
        folder = args.output / 'specializations' / ast_hash / cache_hash
        folder.mkdir(parents=True, exist_ok=False)
        artifacts = []
        for name, original in metadata_group.items():
            path = Path(original)
            if not path.is_file() or name.endswith('.json'):
                continue
            target = folder / path.name
            target.write_bytes(path.read_bytes())
            artifacts.append(dict(path=target.relative_to(args.output).as_posix(),
                kind=path.suffix[1:], sha256=digest(target), bytes=target.stat().st_size))
        assert any(item['kind'] == 'ttir' for item in artifacts)
        assert any(item['kind'] == 'npubin' for item in artifacts)
        entries[cache_hash] = dict(ast_sha256=ast_hash, cache_hash=cache_hash, symbol=src.name,
            first_case=active_case, signature=normalized(src.signature),
            constants=normalized(getattr(src, 'constants', {})),
            compiler_options=normalized({key: metadata[key] for key in option_names if key in metadata}),
            target=normalized(metadata['target']), metadata=normalized(metadata), artifacts=artifacts)
        save(args.output / 'manifest.json', dict(schema='hgkda.ir-capture/v1',
            source_sha256=digest(args.source), compiler_identity=identity,
            entries=list(entries.values()), cases=case_uses,
            status='capturing', scope='Actual executed specializations; compilation and ATen path capture'))

    def capture(kernel, *values, **options):
        compiled = old_run(kernel, *values, **options)
        if compiled is not None and hasattr(compiled, 'metadata'):
            case_uses.append(dict(case=active_case, symbol=kernel.__name__,
                cache_hash=compiled.metadata.hash))
        return compiled

    knobs.compilation.listener = listener
    JITFunction.run = capture
    try:
        for case in cases:
            active_case = case['name']
            inputs = oracle.gen_inputs(case, 'npu:0')
            with CandidateOnly():
                candidate.run(**inputs)
            torch.npu.synchronize()
    finally:
        knobs.compilation.listener = old_listener
        JITFunction.run = old_run
    assert entries and case_uses
    save(args.output / 'manifest.json', dict(schema='hgkda.ir-capture/v1', status='captured',
        source_sha256=digest(args.source), compiler_identity=identity, capture_tool_sha256=digest(__file__),
        entries=list(entries.values()), cases=case_uses,
        scope='Actual executed specializations and ATen path capture; full semantic checks are separate'))
    print(json.dumps(dict(status='captured', cases=len(cases), specializations=len(entries))))


if __name__ == '__main__':
    main()
