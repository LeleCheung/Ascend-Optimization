"""Snapshot one real Triton launch for simulator analysis of later pipeline stages."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def validate(path, source_sha256, case, kernel):
    path = Path(path).resolve()
    data = read(path)
    assert data['source_sha256'] == source_sha256 and data['case'] == case
    assert data['kernel'] == kernel
    tensor_path = (path.parent / data['tensors']).resolve()
    assert tensor_path.is_relative_to(path.parent), 'Snapshot tensor path escapes bundle'
    assert sha(tensor_path) == data['tensors_sha256'], 'Snapshot tensor bytes changed'
    return data, tensor_path


def freeze(path, output, source_sha256, case, kernel):
    data, tensor_path = validate(path, source_sha256, case, kernel)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(tensor_path, output / 'tensors.pt')
    original = dict(manifest=str(Path(path).resolve()), manifest_sha256=sha(path))
    data.update(tensors='tensors.pt', original=original)
    target = output / 'manifest.json'
    target.write_text(json.dumps(data, indent=2) + '\n')
    return dict(manifest=str(target), manifest_sha256=sha(target), original=original)


def snapshot(spec_path, kernel_name, output):
    import torch
    import torch_npu
    from triton.runtime.jit import JITFunction
    from .ir_input import install
    from .paired_driver import imported
    spec = read(spec_path)
    assert sha(spec['source']) == spec['source_sha256']
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    if spec.get('ir_input'):
        install(spec)
    candidate = imported('_snapshot_source', spec['source'])
    oracle = imported('_snapshot_oracle', Path(spec['leaf']) / 'oracle.py')
    inputs = oracle.gen_inputs(spec['case'], 'npu:0')
    capture_call(spec, kernel_name, output, lambda: candidate.run(**inputs), sha(spec_path))


def capture_call(spec, kernel_name, output, invoke, original_spec_sha256):
    """Capture a launch reached by an existing benchmark's unchanged input adapter."""
    import torch
    from triton.runtime.jit import JITFunction
    assert sha(spec['source']) == spec['source_sha256']
    output.mkdir(parents=True, exist_ok=False)
    original = JITFunction.run
    tensors = []
    views = {}
    storages = {}
    launches = []

    class Captured(Exception):
        pass

    def encode(value):
        if isinstance(value, torch.Tensor):
            assert value.is_contiguous() and value.storage_offset() == 0, 'Replay currently supports whole contiguous tensors'
            assert value.numel() * value.element_size() == value.untyped_storage().nbytes()
            storage = value.untyped_storage().data_ptr()
            view = (storage, tuple(value.shape), tuple(value.stride()), str(value.dtype))
            assert storage not in storages or storages[storage] == view, 'Distinct aliased views require explicit replay support'
            storages[storage] = view
            if view not in views:
                views[view] = len(tensors)
                tensors.append(value.detach().cpu())
            return dict(tensor=views[view])
        if isinstance(value, tuple):
            return dict(tuple=[encode(item) for item in value])
        if value is None or isinstance(value, (int, float, bool, str)):
            return value
        raise TypeError('Unsupported launch replay value: ' + str(type(value)))

    def capture(kernel, *values, **options):
        if kernel.__name__ != kernel_name:
            launches.append(kernel.__name__)
            return original(kernel, *values, **options)
        torch.npu.synchronize()
        options = dict(options)
        if callable(options.get('grid')):
            signature = inspect.signature(kernel.fn)
            arguments = signature.bind(*values, **{key: value for key, value in options.items() if key in signature.parameters})
            options['grid'] = tuple(options['grid'](dict(arguments.arguments, **{key: value for key, value in options.items() if key not in arguments.arguments})))
        payload = dict(values=encode(values), options={key: encode(value) for key, value in options.items()})
        torch.save(tensors, output / 'tensors.pt')
        record = dict(source_sha256=spec['source_sha256'], case=spec['case'], kernel=kernel_name,
            tensors='tensors.pt', tensors_sha256=sha(output / 'tensors.pt'), launch=payload,
            preceding_real_launches=launches, original_spec_sha256=original_spec_sha256,
            contract_sha256=spec.get('contract_sha256'),
            ir_input=spec.get('ir_input'), tool_sha256=sha(__file__),
            scope='Analysis-only snapshot immediately before selected launch; preceding kernels ran on real hardware. Target execution and complete-call correctness/performance are verified separately.')
        (output / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
        raise Captured()

    JITFunction.run = capture
    try:
        invoke()
    except Captured:
        pass
    finally:
        JITFunction.run = original
    assert (output / 'manifest.json').exists(), 'Selected launch was not reached'


def validate_replay_ir(data, spec):
    original_binding, replay_binding = data.get('ir_input'), spec.get('ir_input')
    assert bool(original_binding) == bool(replay_binding), 'Replay IR mode differs from the real snapshot'
    if not original_binding:
        return
    assert sha(data['ir_input']['manifest']) == data['ir_input']['manifest_sha256']
    assert sha(spec['ir_input']['manifest']) == spec['ir_input']['manifest_sha256']
    original_ir = read(data['ir_input']['manifest'])
    replay_ir = read(spec['ir_input']['manifest'])
    # Other stages produced the tensors and must retain their exact IR identity.
    expected = {row['ast_sha256']: row['sha256'] for row in original_ir['overrides']
                if row['symbol'] != data['kernel']}
    actual = {row['ast_sha256']: row['sha256'] for row in replay_ir['overrides']
              if row['symbol'] != data['kernel']}
    assert actual == expected, 'Preceding-stage IR differs from the real snapshot'


def load_call(spec, candidate):
    import torch
    from triton.runtime.jit import JITFunction
    binding = spec['kernel_replay']
    path = Path(binding['manifest'])
    assert sha(path) == binding['manifest_sha256']
    data, tensor_path = validate(path, spec['source_sha256'], spec['case'], binding['kernel'])
    if data.get('contract_sha256'):
        assert spec.get('contract_sha256') == data['contract_sha256'], 'Replay contract differs from snapshot'
    validate_replay_ir(data, spec)
    tensors = [value.to('npu:0') for value in torch.load(tensor_path, map_location='cpu', weights_only=True)]
    def decode(value):
        if isinstance(value, dict):
            if set(value) == {'tensor'}:
                return tensors[value['tensor']]
            assert set(value) == {'tuple'}
            return tuple(decode(item) for item in value['tuple'])
        return value
    kernel = getattr(candidate, data['kernel'])
    while not isinstance(kernel, JITFunction):
        kernel = kernel.fn
    return kernel, decode(data['launch']['values']), {key: decode(value) for key, value in data['launch']['options'].items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--kernel-name', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    snapshot(args.spec, args.kernel_name, args.output)


if __name__ == '__main__':
    main()
