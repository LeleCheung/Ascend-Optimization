"""Freeze IR recapture resources while preserving original operator-source identity."""
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def member(root, name):
    path = (root / name).resolve()
    assert path.is_relative_to(root.resolve()), 'IR resource escapes its bundle'
    return path


def freeze(manifest_path, destination, source, runtime_path=None, analysis_target_alias=None):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    original = json.loads(manifest_path.read_bytes())
    assert sha(source) == original['source_sha256']
    assert sha(member(root, original['source'])) == original['source_sha256']
    runtime = root / 'ir_override_runtime.py'
    resources = {original['source']: original['source_sha256'], runtime.name: sha(runtime)}
    for item in original['overrides']:
        assert item['path'] not in resources, 'Duplicate IR resource'
        resources[item['path']] = item['sha256']
    for name, digest in resources.items():
        assert sha(member(root, name)) == digest, 'IR input changed: ' + name
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in resources:
        target = member(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(member(root, name).read_bytes())
    original_runtime_sha256 = resources[runtime.name]
    if runtime_path is not None:
        selected_runtime = Path(runtime_path).resolve()
        (destination / runtime.name).write_bytes(selected_runtime.read_bytes())
        resources[runtime.name] = sha(selected_runtime)
    # Save recapture caches/receipts separately; retain the original evaluation manifest and all compute resources by hash.
    manifest = dict(original, cache_root=str(destination / 'compiler-state'),
        receipt_directory=str(destination / 'compiler-receipts'))
    if analysis_target_alias is not None:
        manifest['analysis_target_alias'] = analysis_target_alias
    target_manifest = destination / 'ir-manifest.json'
    target_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    binding = dict(manifest=str(target_manifest), manifest_sha256=sha(target_manifest),
        original_manifest=str(manifest_path), original_manifest_sha256=sha(manifest_path),
        runtime_sha256=resources[runtime.name], original_runtime_sha256=original_runtime_sha256,
        source_sha256=original['source_sha256'],
        resources=resources)
    if analysis_target_alias is not None:
        binding['analysis_target_alias'] = analysis_target_alias
    return member(destination, original['source']), binding


def binary_bindings(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_bytes())
    expected = {item['ast_sha256']: item for item in manifest['overrides']}
    binaries = {}
    for receipt in Path(manifest['receipt_directory']).glob('*.jsonl'):
        for line in receipt.read_text().splitlines():
            row = json.loads(line)
            assert row['manifest_sha256'] == sha(path)
            assert row['source_sha256'] == manifest['source_sha256']
            if not row['override_applied']:
                continue
            key = row['ast_sha256']
            assert row['ir_sha256'] == expected[key]['sha256']
            artifacts = [item for item in row['artifacts'] if item['kind'] == 'npubin']
            assert len(artifacts) == 1 and sha(artifacts[0]['path']) == artifacts[0]['sha256']
            digest = artifacts[0]['sha256']
            assert key not in binaries or binaries[key] == digest
            binaries[key] = digest
    assert set(binaries) == set(expected), 'Real-device binary evidence missing for an override'
    return binaries


def install(spec):
    binding = spec.get('ir_input')
    if not binding:
        return
    path = Path(binding['manifest']).resolve()
    root = path.parent
    assert sha(path) == binding['manifest_sha256'], 'IR collection manifest changed'
    assert binding['source_sha256'] == spec['source_sha256'] == sha(spec['source'])
    for name, digest in binding['resources'].items():
        assert sha(member(root, name)) == digest, 'IR collection resource changed: ' + name
    runtime = root / 'ir_override_runtime.py'
    assert sha(runtime) == binding['runtime_sha256']
    module_spec = importlib.util.spec_from_file_location('_hgkda_ir_collection_runtime', runtime)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module.install(path).resolve() == Path(spec['source']).resolve()
