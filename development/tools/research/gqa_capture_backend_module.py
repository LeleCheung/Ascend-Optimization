"""Capture the actual temporary HIVM backend input from an unchanged compile."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from gqa_binary_execution import execution_signature
from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = read(args.reference_request)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    temp = out / 'tmp'
    temp.mkdir()
    command = list(request['command'])
    assert Path(command[0]).name == 'bishengir-compile'
    assert command.count('-o') == 1
    command[command.index('-o') + 1] = str(out / 'kernel')
    command = [arg for arg in command if arg not in {
        '--mlir-print-ir-after-all', '--mlir-print-ir-module-scope'}]
    save(out / 'request.json', dict(command=command, TMPDIR=str(temp),
        reference_request_sha256=sha(args.reference_request),
        expected_binary=request['expected_binary'],
        scope='Unchanged compilation. Capture compiler-created backend input; no candidate edits.'))
    captured = {}
    with (out / 'stdout.txt').open('x') as stdout, (out / 'stderr.txt').open('x') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   env=dict(os.environ, TMPDIR=str(temp)))
        while True:
            # Replace partial snapshots as the compiler finishes writing the file.
            for path in temp.rglob('*.hivm.opt.mlir'):
                try:
                    data = path.read_bytes()
                except FileNotFoundError:
                    continue
                (out / path.name).write_bytes(data)
                captured[path.name] = len(data)
            if process.poll() is not None:
                break
            time.sleep(0.005)
    binary = out / 'kernel.o'
    expected = Path(request['expected_binary'])
    matches = (binary.exists() and
               execution_signature(binary) == execution_signature(expected))
    result = dict(returncode=process.returncode, captured=captured,
                  execution_matches=matches,
                  binary_sha256=sha(binary) if binary.exists() else None,
                  expected_binary_sha256=sha(expected))
    save(out / 'result.json', result)
    print(result)
    assert process.returncode == 0 and matches and captured


if __name__ == '__main__':
    main()
