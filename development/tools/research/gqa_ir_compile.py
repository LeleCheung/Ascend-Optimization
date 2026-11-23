"""Check model-authored GQA BCMLIR feasibility with the original backend options."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ir', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    import torch_npu
    from triton.backends.ascend.compiler import NPUOptions, _compile_linalg_to_npu_bin
    metadata = json.loads(a.metadata.read_bytes())
    names = {field.name for field in fields(NPUOptions)}
    options = NPUOptions(**{key: value for key, value in metadata.items() if key in names})
    a.output.mkdir(parents=True, exist_ok=False)
    record = dict(ir=str(a.ir), ir_sha256=sha(a.ir), metadata=str(a.metadata),
        metadata_sha256=sha(a.metadata), tool_sha256=sha(__file__),
        scope='Backend compilation feasibility only. Four-case correctness and full-call performance remain separate.')
    try:
        binary = _compile_linalg_to_npu_bin(a.ir.read_text(), metadata, options)
        target = a.output / '_gqa_fwd_kernel.npubin'
        target.write_bytes(binary)
        record.update(status='compiled', binary_sha256=sha(target), bytes=len(binary))
    except subprocess.CalledProcessError as error:
        (a.output / 'compiler-stdout.txt').write_bytes(error.stdout or b'')
        (a.output / 'compiler-stderr.txt').write_bytes(error.stderr or b'')
        record.update(status='compile_failed', returncode=error.returncode, command=error.cmd)
    (a.output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
