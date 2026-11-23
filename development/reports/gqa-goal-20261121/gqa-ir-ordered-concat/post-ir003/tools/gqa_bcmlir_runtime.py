"""Apply one exact source-bound GQA BCMLIR replacement inside its unchanged harness."""
import hashlib
import json
import os
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_input(incoming, manifest):
    result = incoming.replace(str(Path(manifest['source'])), manifest['original_source_location'])
    for alias, recorded in manifest.get('target_aliases', {}).items():
        result = result.replace('#hacc.target<"' + alias + '">', '#hacc.target<"' + recorded + '">')
    return result


def install(manifest_path):
    import torch
    import torch_npu
    from triton import knobs
    import triton.backends.ascend.compiler as backend
    manifest_path = Path(manifest_path).resolve()
    m = json.loads(manifest_path.read_bytes())
    for name in ['source', 'contract', 'original_ir', 'changed_ir']:
        assert sha(m[name]) == m[name + '_sha256'], name + ' identity changed'
    cache = Path(m['cache']) / sha(manifest_path)
    os.environ['TRITON_CACHE_DIR'] = str(cache)
    knobs.cache.dir = str(cache)
    original_ir = Path(m['original_ir']).read_text()
    changed_ir = Path(m['changed_ir']).read_text()
    original_compile = backend._compile_linalg_to_npu_bin
    receipt_path = Path(m['receipts'])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    def record_compile(incoming, applied, metadata, matches, succeeded, binary):
        row = dict(manifest_sha256=sha(manifest_path), source_sha256=m['source_sha256'],
            contract_sha256=m['contract_sha256'], symbol=metadata.get('kernel_name'),
            incoming_ir_sha256=hashlib.sha256(incoming.encode()).hexdigest(),
            compiled_ir_sha256=hashlib.sha256(applied.encode()).hexdigest(),
            override_applied=matches, compiled=succeeded,
            binary_sha256=hashlib.sha256(binary).hexdigest() if binary else None)
        with receipt_path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
    def wrapped(incoming, metadata, options):
        # Normalize copied source locations and explicitly registered device
        # aliases. Every other byte must match the recorded target lowering.
        canonical = canonical_input(incoming, m)
        matches = canonical == original_ir
        # The backend fills metadata.kernel_name while compiling. Inspect the
        # actual function declaration before compilation to catch mismatches.
        target_symbol = re.search(r'func\.func\s+@' + re.escape(m['symbol']) + r'\(', incoming)
        if target_symbol and m['target_marker'] in incoming:
            if not matches:
                (receipt_path.parent / 'unexpected-target.bcmlir').write_text(incoming)
                raise AssertionError('Target lowering differs beyond source locations')
        applied = changed_ir if matches else incoming
        try:
            binary = original_compile(applied, metadata, options)
        except Exception:
            record_compile(incoming, applied, metadata, matches, False, None)
            raise
        record_compile(incoming, applied, metadata, matches, True, binary)
        return binary
    backend._compile_linalg_to_npu_bin = wrapped
    return m
