"""Separate enqueue and final-drain wall time without changing formal GQA results."""
import argparse
import cProfile
import fcntl
import importlib.util
from pathlib import Path
import pstats
import shlex
import statistics
import subprocess
import time

from ir_campaign import read, save, sha


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(root):
    import torch
    import torch_npu
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    protocol = read(root / 'protocol.json')
    assert sha(protocol['contract']) == protocol['contract_sha256']
    contract = read(protocol['contract'])
    modules = {}
    for label, source in protocol['sources'].items():
        assert sha(source['path']) == source['sha256']
        modules[label] = load(source['path'], 'host_diagnostic_' + label)
    rows = []
    for case_id in protocol['case_ids']:
        case = next(row for row in contract['cases'] if row['case_id'] == case_id)
        dtype = getattr(torch, case['dtype'][0])
        gen = torch.Generator(device='cpu').manual_seed(1729 + case_id)
        inputs = [(torch.rand(shape, generator=gen)*2-1).to(dtype).to('npu:0')
                  for shape in case['input_shape']]
        attrs = case['attrs']
        scale = attrs['scaleValue']
        if scale <= 0:
            scale = inputs[0].shape[-1] ** -.5
        sq, sk = inputs[0].shape[1], inputs[1].shape[1]
        mask = ((torch.arange(sk)[None, :] > torch.arange(sq)[:, None] + sk-sq).to('npu:0')
                if attrs['is_causal'] else None)

        def native():
            return torch.ops.npu.npu_fused_infer_attention_score(
                *inputs, num_heads=inputs[0].shape[2], num_key_value_heads=inputs[1].shape[2],
                input_layout='BSND', scale=scale, atten_mask=mask, sparse_mode=0)[0]

        functions = {'native': native}
        functions.update({label: (lambda fn=module.run: fn(*inputs, **attrs))
                          for label, module in modules.items()})
        for label, fn in functions.items():
            for _ in range(contract['warmup']):
                fn()
            torch.npu.synchronize()
            trials = []
            n = contract['iterations']
            for _ in range(contract['trials']):
                start = time.perf_counter_ns()
                for _ in range(n):
                    fn()
                submitted = time.perf_counter_ns()
                torch.npu.synchronize()
                done = time.perf_counter_ns()
                trials.append(dict(enqueue_us_per_call=(submitted-start)/1000/n,
                                   final_drain_us_per_call=(done-submitted)/1000/n,
                                   total_us_per_call=(done-start)/1000/n))
            profiler = cProfile.Profile()
            torch.npu.synchronize()
            profiler.enable()
            for _ in range(n):
                fn()
            profiler.disable()
            torch.npu.synchronize()
            profile_file = root / ('case' + str(case_id) + '-' + label + '.prof')
            profiler.dump_stats(str(profile_file))
            stats = pstats.Stats(profiler)
            host_calls = [dict(file=key[0], line=key[1], function=key[2], primitive_calls=v[0],
                               total_calls=v[1], self_us_per_invocation=v[2]*1e6/n,
                               cumulative_us_per_invocation=v[3]*1e6/n)
                          for key, v in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:45]]
            row = dict(case_id=case_id, label=label, trials=trials,
                       medians={key: statistics.median(t[key] for t in trials) for key in trials[0]},
                       cprofile_top=host_calls)
            rows.append(row)
            save(root / ('case' + str(case_id) + '-' + label + '.json'), row)
            print({k: row[k] for k in ['case_id', 'label', 'medians']}, flush=True)
    save(root / 'summary.json', dict(status='completed', protocol_sha256=sha(root/'protocol.json'),
                                    rows=rows, scope=protocol['scope']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--invocations', nargs='+')
    parser.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.execute:
        execute(args.execute.resolve())
        return
    run = args.run.resolve()
    out = run / 'host-timing-diagnostic' / args.name
    out.mkdir(parents=True, exist_ok=False)
    config = read(run / 'config.json')
    sources = {}
    for invocation in args.invocations:
        source = run / 'agent-calls' / invocation / 'output-source.py'
        result = read(run / 'iterations' / invocation / 'parent-validation/results.json')
        assert result['source_sha256'] == sha(source)
        assert result['contract_sha256'] == sha(run / 'contract.json')
        assert {r['case_id'] for r in result['cases']} == {1, 2, 4, 7}
        assert all(r['status'] == 'measured' and all(
            r[side+'_correctness']['official_checker']['passed'] for side in ['native', 'candidate'])
            for r in result['cases'])
        sources[invocation] = dict(path=str(source), sha256=sha(source))
    save(out/'protocol.json', dict(contract=str(run/'contract.json'), contract_sha256=sha(run/'contract.json'),
        sources=sources, case_ids=[4, 1],
        reason='Similar full-call costs across sizes and conflicting hardware versus full-call improvements warrant host dispatch diagnosis.',
        scope='Diagnostic only, not a new formal timing result or correctness test. Original inputs, seeds, warmup, trials and iteration counts. '
              'Enqueue wall time can include runtime blocking/device backpressure; drain is remaining synchronization, not total device work. '
              'cProfile is a separate instrumented batch, never a speed score. Formal contract and results remain unchanged.'))
    shell = 'source ' + shlex.quote(config['sdk']+'/set_env.sh') + '\n'
    for key, value in dict(ASCEND_RT_VISIBLE_DEVICES=str(config['device']), OMP_NUM_THREADS='1',
                          TRITON_CACHE_DIR=str(out/'cache'), PYTHONDONTWRITEBYTECODE='1',
                          TRITON_DISABLE_LINE_INFO='false').items():
        shell += 'export '+key+'='+shlex.quote(value)+'\n'
    shell += 'exec '+shlex.join([config['device_python'], str(Path(__file__).resolve()), '--execute', str(out)])
    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (out/'execution.log').open('x') as stream:
            process = subprocess.run(['docker', 'exec', '-w', str(out), config['container'], 'bash', '-c', shell],
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
    save(out/'execution.json', dict(returncode=process.returncode))
    assert process.returncode == 0


if __name__ == '__main__':
    main()
