"""Capture compiler pass output for an unchanged, measured GQA BCMLIR input."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import re
import subprocess

from ir_campaign import save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ir', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-binary', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu
    from triton.backends.ascend.compiler import NPUOptions, _compile_linalg_to_npu_bin
    from gqa_binary_execution import execution_signature
    metadata = json.loads(args.metadata.read_bytes())
    option_names = {field.name for field in fields(NPUOptions)}
    options = NPUOptions(**{key: value for key, value in metadata.items() if key in option_names})
    original_run = subprocess.run
    invocations = []
    def capture(command, *positional, **keywords):
        if isinstance(command, (list, tuple)) and 'bishengir-compile' in str(command[0]):
            command = [*command, '--mlir-disable-threading', '--mlir-print-ir-after-all',
                       '--mlir-print-ir-module-scope']
            process = original_run(command, *positional, **keywords)
            text = (process.stdout or b'') + (process.stderr or b'')
            path = args.output / ('compiler-passes-' + str(len(invocations)) + '.txt')
            path.write_bytes(text)
            invocations.append(dict(command=command, returncode=process.returncode, dump=path.name,
                dump_sha256=sha(path), stages=re.findall(r'IR Dump After ([^\n]+)', text.decode('utf8', 'replace'))))
            return process
        return original_run(command, *positional, **keywords)
    result = dict(ir_sha256=sha(args.ir), metadata_sha256=sha(args.metadata),
                  expected_binary_sha256=sha(args.expected_binary),
                  scope='Compiler diagnostics on unchanged measured IR. No source, IR or candidate edit.')
    try:
        subprocess.run = capture
        binary = _compile_linalg_to_npu_bin(args.ir.read_text(), metadata, options)
        path = args.output / 'unchanged-input.npubin'
        path.write_bytes(binary)
        assert invocations, 'No compiler pass dump captured'
        same_execution = execution_signature(path) == execution_signature(args.expected_binary)
        result.update(status='captured', binary_sha256=sha(path), execution_matches_measured=same_execution)
        assert same_execution, 'Diagnostic compile differs from measured execution; investigate before using dump'
    except Exception as error:
        detail = repr(error)
        (args.output / 'error-detail.txt').write_text(detail, encoding='utf8')
        result.update(status='capture_failed', error=detail[:500], error_detail='error-detail.txt')
        if isinstance(error, subprocess.CalledProcessError):
            (args.output / 'error-stdout.txt').write_bytes(error.stdout or b'')
            (args.output / 'error-stderr.txt').write_bytes(error.stderr or b'')
    finally:
        subprocess.run = original_run
        result['invocations'] = invocations
        save(args.output / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'invocations'}))


if __name__ == '__main__':
    main()
