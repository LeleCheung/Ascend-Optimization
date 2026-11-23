"""Conditional ABBA measurement of the existing GQA target, using frozen timing parameters."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

from ir_campaign import read, save, sha


def execute(root):
    import torch
    import torch_npu
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    protocol = read(root / 'protocol.json')
    contract = read(Path(protocol['contract']))
    assert sha(protocol['contract']) == protocol['contract_sha256']
    case = next(row for row in contract['cases'] if row['case_id'] == 4)
    generator = torch.Generator(device='cpu').manual_seed(1729 + case['case_id'])
    dtype = getattr(torch, case['dtype'][0])
    inputs = [((torch.rand(shape, generator=generator)*2-1).to(dtype)).to('npu:0') for shape in case['input_shape']]
    functions = {}
    for label, row in protocol['sources'].items():
        source = Path(row['path'])
        assert sha(source) == row['sha256']
        spec = importlib.util.spec_from_file_location('gqa_stability_'+label, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        functions[label] = module.run
        for _ in range(contract['warmup']):
            module.run(*inputs, **case['attrs'])
        torch.npu.synchronize()
    records = []
    for index, label in enumerate(protocol['order']):
        function = functions[label]
        for _ in range(contract['warmup']):
            function(*inputs, **case['attrs'])
        torch.npu.synchronize()
        trials = []
        for _ in range(contract['trials']):
            start = time.perf_counter_ns()
            for _ in range(contract['iterations']):
                function(*inputs, **case['attrs'])
            torch.npu.synchronize()
            trials.append((time.perf_counter_ns()-start)/1000/contract['iterations'])
        record = dict(index=index, label=label, trials_us=trials, median_us=statistics.median(trials))
        save(root / ('block-'+str(index)+'.json'), record)
        records.append(record)
    original = [r['median_us'] for r in records if r['label']=='A']
    candidate = [r['median_us'] for r in records if r['label']=='B']
    result = dict(status='completed', records=records, original_median_us=statistics.median(original),
        candidate_median_us=statistics.median(candidate), both_candidate_blocks_faster=max(candidate)<min(original),
        latency_reduction_pct=(1-statistics.median(candidate)/statistics.median(original))*100,
        contract_sha256=protocol['contract_sha256'], source_identities=protocol['sources'], target_case=4,
        scope='Conditional timing check only. Existing target input, seed, warmup, trials and batch iterations; '
              'full four-case official correctness remains in the referenced formal results. '
              'Both modules compile/warm before measurement; one device lock covers all blocks.')
    save(root / 'summary.json', result)
    print(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--invocation')
    parser.add_argument('--baseline-source', type=Path,
                        help='Qualified same-contract comparison source when resolving sibling candidates')
    parser.add_argument('--reason', default='A specific measurement concern requires a same-environment timing check.')
    parser.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.execute:
        execute(args.execute.resolve())
        return
    run = args.run.resolve()
    config = read(run / 'config.json')
    out = run / 'conditional-stability' / args.invocation
    out.mkdir(parents=True, exist_ok=False)
    source_parent = run / 'agent-calls' / args.invocation / 'input-source.py'
    parent = args.baseline_source.resolve() if args.baseline_source else source_parent
    if args.baseline_source:
        assert parent.is_relative_to(run)
        matches = [read(path) for path in (run / 'iterations').glob('*/parent-validation/results.json')
                   if read(path).get('source_sha256') == sha(parent)]
        assert any(result['contract_sha256'] == sha(run / 'contract.json')
                   and {row['case_id'] for row in result['cases']} == {1, 2, 4, 7}
                   and all(row.get('status') == 'measured' and all(
                       row.get(side+'_correctness', {}).get('official_checker', {}).get('passed')
                       for side in ['candidate', 'native']) for row in result['cases']) for result in matches)
    candidate = run / 'agent-calls' / args.invocation / 'output-source.py'
    result = run / 'iterations' / args.invocation / 'parent-validation/results.json'
    assert read(result)['source_sha256'] == sha(candidate)
    protocol = dict(reason=args.reason,
        registered_at=datetime.now(timezone.utc).isoformat(), order=['A','B','B','A'], target_case=4,
        contract=str(run / 'contract.json'), contract_sha256=sha(run / 'contract.json'),
        sources=dict(A=dict(path=str(parent),sha256=sha(parent)),B=dict(path=str(candidate),sha256=sha(candidate))),
        full_candidate_result=str(result), full_candidate_result_sha256=sha(result),
        source_parent_sha256=sha(source_parent),
        comparison_baseline_kind='qualified_sibling' if args.baseline_source else 'source_parent',
        decision_rule='One conditional comparison; retain only if both B-block medians are faster than both A blocks.')
    save(out / 'protocol.json', protocol)
    shell = 'source '+shlex.quote(config['sdk']+'/set_env.sh')+'\n'
    for key, value in dict(ASCEND_RT_VISIBLE_DEVICES=str(config['device']), OMP_NUM_THREADS='1',
                          TRITON_CACHE_DIR=str(out/'cache'), PYTHONDONTWRITEBYTECODE='1',
                          TRITON_DISABLE_LINE_INFO='false').items():
        shell += 'export '+key+'='+shlex.quote(value)+'\n'
    shell += 'exec '+shlex.join([config['device_python'],str(Path(__file__).resolve()),'--execute',str(out)])
    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (out/'execution.log').open('x') as stream:
            process = subprocess.run(['docker','exec','-w',str(out),config['container'],'bash','-c',shell],
                                     stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
    save(out/'execution.json',dict(returncode=process.returncode))
    assert process.returncode==0
    print(read(out/'summary.json'))


if __name__=='__main__':
    main()
