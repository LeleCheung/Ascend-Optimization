"""Check warm-launch lifecycle on existing GQA inputs; never produce timing scores."""
import argparse
import fcntl
from pathlib import Path
import shlex
import subprocess

from gqa_host_timing_diagnostic import load
from ir_campaign import read, save, sha


def execute(out):
    import torch
    import torch_npu
    from triton.backends.ascend.driver import NPULauncher
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    protocol = read(out/'protocol.json')
    assert sha(protocol['source']) == protocol['source_sha256']
    assert sha(protocol['contract']) == protocol['contract_sha256']
    contract = read(protocol['contract'])
    module = load(protocol['source'], 'cache_lifecycle_candidate')
    kernel = module._gqa_fwd_kernel
    original_jit, original_launch = kernel.run, NPULauncher.__call__
    jit_calls, launches = [], []

    def counted_jit(*args, **kwargs):
        jit_calls.append(1)
        return original_jit(*args, **kwargs)

    def counted_launch(self, *args, **kwargs):
        launches.append(dict(stream=args[3], pointers=[x.data_ptr() for x in args[9:13]]))
        return original_launch(self, *args, **kwargs)

    kernel.run = counted_jit
    NPULauncher.__call__ = counted_launch
    records = []
    try:
        for case_id in [4, 1]:
            case = next(c for c in contract['cases'] if c['case_id'] == case_id)
            generator = torch.Generator(device='cpu').manual_seed(1729+case_id)
            inputs = [(torch.rand(s, generator=generator)*2-1).to(getattr(torch, case['dtype'][0])).to('npu:0')
                      for s in case['input_shape']]
            torch.npu.synchronize()
            outputs = []

            def invoke(label, tensors, expected_jit_calls):
                j0, l0 = len(jit_calls), len(launches)
                expected_stream = torch_npu._C._npu_getCurrentRawStreamNoWait(0)
                output = module.run(*tensors, **case['attrs'])
                torch.npu.synchronize()
                row = dict(case_id=case_id, label=label, jit_calls=len(jit_calls)-j0,
                           launches=len(launches)-l0, stream=expected_stream,
                           expected_pointers=[t.data_ptr() for t in tensors]+[output.data_ptr()],
                           observed=launches[l0:], output_unique=all(output.data_ptr()!=t.data_ptr() for t in outputs),
                           equal_cold_output=not outputs or bool(torch.equal(output, outputs[0])))
                assert row['jit_calls'] == expected_jit_calls, row
                assert row['launches'] == 1 and row['output_unique'] and row['equal_cold_output'], row
                assert row['observed'][0] == dict(stream=expected_stream, pointers=row['expected_pointers']), row
                outputs.append(output)
                records.append(row)
                save(out/'partial.json', records)

            invoke('cold', inputs, 1)
            invoke('warm', inputs, 0)
            fresh_inputs = [t.clone() for t in inputs]
            torch.npu.synchronize()
            assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(inputs,fresh_inputs))
            invoke('fresh_addresses', fresh_inputs, 0)
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                invoke('current_stream', fresh_inputs, 0)
            seen = []
            def hook(*args, **kwargs):
                seen.append(1)
            kernel.pre_run_hooks.append(hook)
            try:
                invoke('active_hook_fallback', inputs, 1)
            finally:
                kernel.pre_run_hooks.remove(hook)
            assert len(seen) == 1
            invoke('warm_after_hook', inputs, 0)
    finally:
        kernel.run = original_jit
        NPULauncher.__call__ = original_launch
    save(out/'summary.json', dict(status='passed', records=records, protocol_sha256=sha(out/'protocol.json'),
                                 scope=protocol['scope']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--invocation')
    parser.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.execute:
        execute(args.execute.resolve())
        return
    run = args.run.resolve()
    out = run/'host-timing-diagnostic'/('lifecycle-'+args.invocation)
    out.mkdir(parents=True, exist_ok=False)
    source = run/'agent-calls'/args.invocation/'output-source.py'
    config = read(run/'config.json')
    save(out/'protocol.json', dict(source=str(source), source_sha256=sha(source),
        contract=str(run/'contract.json'), contract_sha256=sha(run/'contract.json'),
        reason='First compiled-launch cache requires proof of actual hits, one launch per call, fresh pointers, dynamic stream and hook fallback.',
        scope='One targeted host lifecycle diagnostic on existing case4/1 seed inputs. No new shapes, numerical thresholds or timing scores; originalfour remains formal correctness.'))
    shell = 'source '+shlex.quote(config['sdk']+'/set_env.sh')+'\n'
    for key, value in dict(ASCEND_RT_VISIBLE_DEVICES=str(config['device']), OMP_NUM_THREADS='1',
                          TRITON_CACHE_DIR=str(out/'cache'), PYTHONDONTWRITEBYTECODE='1').items():
        shell += 'export '+key+'='+shlex.quote(value)+'\n'
    shell += 'exec '+shlex.join([config['device_python'],str(Path(__file__).resolve()),'--execute',str(out)])
    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (out/'execution.log').open('x') as log:
            p = subprocess.run(['docker','exec','-w',str(out),config['container'],'bash','-c',shell],
                               stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    save(out/'execution.json', dict(returncode=p.returncode))
    assert p.returncode == 0


if __name__ == '__main__':
    main()
