"""Install process-local, source-bound TTIR overrides and record compiler receipts.

This generic loader changes compiler input selection, never operator source or a
system installation. The uploaded manifest and TTIR files are immutable inputs.
"""
import base64
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(value):
    """Keep tuple-keyed compiler constants serializable without losing keys."""
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: normalized(item) for key, item in sorted(value.items())}
        pairs = [[normalized(key), normalized(item)] for key, item in value.items()]
        pairs.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True))
        return {'__mapping__': pairs}
    if isinstance(value, tuple):
        return {'__tuple__': [normalized(item) for item in value]}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return json.loads(json.dumps(value, default=str, sort_keys=True))


def compiler_identity():
    import triton
    root = Path(triton.__file__).parent
    sources = ['compiler/compiler.py', 'backends/ascend/compiler.py', 'runtime/cache.py', 'knobs.py']
    excluded = {'TRITON_CACHE_DIR', 'TRITON_OVERRIDE_DIR', 'TRITON_DUMP_DIR',
        'TRITON_KERNEL_OVERRIDE', 'TRITON_KERNEL_DUMP', 'TRITON_ALWAYS_COMPILE', 'TRITON_HOME'}
    safe = lambda name: not any(part in name for part in ['TOKEN', 'SECRET', 'PASSWORD', 'AUTH', 'API_KEY'])
    environment = {name: value for name, value in os.environ.items()
        if name.startswith(('TRITON_', 'FLAGTREE_', 'TLE_REPLACE_IR')) and name not in excluded and safe(name)}
    return dict(triton_version=triton.__version__, sources={name: digest(root / name) for name in sources},
        compiler_environment=environment)


def install(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    source = root / manifest['source']
    assert digest(source) == manifest['source_sha256'], 'Frozen source changed'
    assert compiler_identity() == manifest['compiler_identity'], 'Compiler identity or environment changed'
    cache_root = Path(manifest['cache_root']) / manifest_sha
    overrides = cache_root / 'overrides'
    receipt_root = Path(manifest['receipt_directory'])
    receipt_root.mkdir(parents=True, exist_ok=True)
    expected = {}
    for item in manifest['overrides']:
        assert item['filename'].endswith('.ttir'), 'This loader supports native TTIR parsing'
        path = (root / item['path']).resolve()
        assert path.is_relative_to(root)
        assert digest(path) == item['sha256']
        key = item['ast_sha256']
        assert key not in expected, 'One compiler binding per AST specialization is required'
        expected[key] = item
        encoded = base64.b32encode(bytes.fromhex(key)).decode().rstrip('=')
        destination = overrides / encoded / item['filename']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert digest(destination) == item['sha256']
        else:
            destination.write_bytes(path.read_bytes())
    os.environ['TRITON_CACHE_DIR'] = str(cache_root / 'cache')
    os.environ['TRITON_OVERRIDE_DIR'] = str(overrides)
    os.environ['TRITON_KERNEL_OVERRIDE'] = '1'
    from triton import knobs
    knobs.cache.dir = str(cache_root / 'cache')
    knobs.cache.override_dir = str(overrides)
    knobs.compilation.override = True
    old_listener = knobs.compilation.listener

    def listener(src, metadata, metadata_group, times, cache_hit):
        if old_listener:
            old_listener(src=src, metadata=metadata, metadata_group=metadata_group,
                times=times, cache_hit=cache_hit)
        ast_hash = src.hash()
        item = expected.get(ast_hash)
        receipt = dict(schema='hgkda.ir-override-receipt/v1', manifest_sha256=manifest_sha,
            source_sha256=manifest['source_sha256'], ast_sha256=ast_hash,
            cache_hash=metadata['hash'], symbol=src.name, cache_hit=cache_hit,
            override_applied=item is not None)
        if item:
            actual_options = normalized({key: metadata.get(key) for key in item['compiler_options']})
            assert actual_options == item['compiler_options'], 'Compiler options differ from captured IR'
            assert normalized(metadata['target']) == item['target'], 'Compiler target changed'
            actual = Path(metadata_group[item['filename']])
            assert digest(actual) == item['sha256'], 'Override did not reach the cached compiler stage'
            receipt.update(ir_sha256=digest(actual), original_ir_sha256=item['original_sha256'],
                compiler_options=actual_options, target=item['target'])
        receipt['artifacts'] = [dict(kind=Path(name).suffix[1:], path=str(path), sha256=digest(path))
            for name, path in metadata_group.items()
            if Path(path).is_file() and Path(name).suffix in {'.ttir', '.ttadapter', '.bcmlir', '.npubin'}]
        path = receipt_root / (str(os.getpid()) + '.jsonl')
        with path.open('a', encoding='utf8') as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False) + '\n')

    knobs.compilation.listener = listener
    return source
