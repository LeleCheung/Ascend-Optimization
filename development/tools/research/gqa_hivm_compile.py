"""Compile an exported HIVM module with a previously validated backend invocation."""
import argparse
import json
from pathlib import Path
import subprocess

from gqa_binary_execution import execution_signature
from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ir', type=Path, required=True)
    parser.add_argument('--reference-request', type=Path, required=True)
    parser.add_argument('--reference-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    command = list(read(args.reference_request)['command'])
    assert Path(command[0]).name == 'hivmc' and command.count('-o') == 1
    command[1] = str(args.ir.resolve())
    command[command.index('-o') + 1] = str((args.output / 'kernel').resolve())
    record = dict(command=command, ir=str(args.ir.resolve()), ir_sha256=sha(args.ir),
                  reference_request_sha256=sha(args.reference_request),
                  reference_binary_sha256=sha(args.reference_binary),
                  tool_sha256=sha(__file__),
                  scope='Single-operator HIVM backend compile. Runtime binding validates original BCMLIR, source, contract and baseline execution.')
    save(args.output / 'request.json', record)
    with (args.output / 'stdout.txt').open('x') as stdout, (args.output / 'stderr.txt').open('x') as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr)
    binary = args.output / 'kernel.o'
    record.update(returncode=process.returncode, status='compile_failed')
    if process.returncode == 0 and binary.is_file():
        signature = execution_signature(binary)
        reference = execution_signature(args.reference_binary)
        record.update(status='compiled', binary=str(binary.resolve()), binary_sha256=sha(binary),
                      execution_matches_reference=signature == reference,
                      public_symbol_names_match=[s['name'] for s in signature['public_symbols']]
                          == [s['name'] for s in reference['public_symbols']],
                      kernel_arg_metadata_matches=signature['sections'].get('__CCE_KernelArgSize')
                          == reference['sections'].get('__CCE_KernelArgSize'))
        assert record['public_symbol_names_match'] and record['kernel_arg_metadata_matches'], 'HIVM candidate ABI differs'
    save(args.output / 'result.json', record)
    print(json.dumps({key: record[key] for key in ['status', 'returncode', 'ir_sha256']}))


if __name__ == '__main__':
    main()
