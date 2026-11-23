"""Resolve a specific fixed-source GQA IR timing ambiguity under the frozen contract."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

from ir_campaign import read, save, sha


def execute(root, index):
    import torch
    import torch_npu
    from gqa_bcmlir_runtime import install
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    protocol = read(root/'protocol.json')
    label = protocol['order'][index]
    entry = protocol['variants'][label]
    assert sha(entry['manifest']) == entry['manifest_sha256']
    manifest = install(entry['manifest'])
    contract = read(manifest['contract'])
    case = next(row for row in contract['cases'] if row['case_id']==4)
    gen = torch.Generator(device='cpu').manual_seed(1729+4)
    inputs = [(torch.rand(s,generator=gen)*2-1).to(getattr(torch,case['dtype'][0])).to('npu:0')
              for s in case['input_shape']]
    spec = importlib.util.spec_from_file_location('gqa_ir_stability',manifest['source'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for _ in range(contract['warmup']):
        module.run(*inputs,**case['attrs'])
    torch.npu.synchronize()
    observed = {hashlib.sha256(kernel.asm['npubin']).hexdigest()
                for cache in module._gqa_fwd_kernel.device_caches.values()
                for kernel in cache[0].values()}
    assert observed == {manifest['compiled_artifact']['binary_sha256']}, 'Warm executable differs from formal IR artifact'
    trials = []
    for _ in range(contract['trials']):
        start = time.perf_counter_ns()
        for _ in range(contract['iterations']):
            module.run(*inputs,**case['attrs'])
        torch.npu.synchronize()
        trials.append((time.perf_counter_ns()-start)/1000/contract['iterations'])
    save(root/('block-'+str(index)+'.json'),dict(label=label,index=index,trials_us=trials,
        median_us=statistics.median(trials),binary_sha256=next(iter(observed)),
        source_sha256=manifest['source_sha256'],contract_sha256=manifest['contract_sha256']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--name')
    parser.add_argument('--reason')
    parser.add_argument('--execute',type=Path)
    parser.add_argument('--index',type=int)
    args = parser.parse_args()
    if args.execute:
        execute(args.execute.resolve(),args.index)
        return
    out = args.run.resolve()/'conditional-stability'/args.name
    out.mkdir(parents=True,exist_ok=False)
    variants = {}
    identities = set()
    for label,path in [('A',args.baseline),('B',args.candidate)]:
        path = path.resolve()
        m = read(path)
        result = read(path.parent/'results.json')
        assert result['source_sha256']==m['source_sha256'] and result['contract_sha256']==m['contract_sha256']
        assert {r['case_id'] for r in result['cases']}=={1,2,4,7}
        assert all(r['status']=='measured' and all(r[s+'_correctness']['official_checker']['passed']
                   for s in ['candidate','native']) for r in result['cases'])
        identities.add((m['source_sha256'],m['contract_sha256']))
        variants[label] = dict(manifest=str(path),manifest_sha256=sha(path),formal_result_sha256=sha(path.parent/'results.json'))
    assert len(identities)==1 and args.reason
    protocol = dict(reason=args.reason,registered_at=datetime.now(timezone.utc).isoformat(),
        variants=variants,order=['A','B','B','A'],target_case=4,
        scope='One conditional comparison using original target input/seed and timing counts. Separate subprocess per block isolates compiler override state; one device lock covers all blocks. Each block warms first and validates its executable. Formal four-case results remain frozen.')
    save(out/'protocol.json',protocol)
    config = read(args.run/'config.json')
    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for index in range(4):
            shell='source '+shlex.quote(config['sdk']+'/set_env.sh')+'\n'
            for key,value in dict(ASCEND_RT_VISIBLE_DEVICES=str(config['device']),OMP_NUM_THREADS='1',
                                 TRITON_DISABLE_LINE_INFO='false',PYTHONDONTWRITEBYTECODE='1').items():
                shell+='export '+key+'='+shlex.quote(value)+'\n'
            shell+='exec '+shlex.join([config['device_python'],str(Path(__file__).resolve()),
                                      '--execute',str(out),'--index',str(index)])
            with (out/('block-'+str(index)+'.log')).open('x') as log:
                p=subprocess.run(['docker','exec',config['container'],'bash','-c',shell],
                                 stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            save(out/('execution-'+str(index)+'.json'),dict(returncode=p.returncode))
            assert p.returncode==0
    records=[read(out/('block-'+str(i)+'.json')) for i in range(4)]
    a=[r['median_us'] for r in records if r['label']=='A']
    b=[r['median_us'] for r in records if r['label']=='B']
    result=dict(status='completed',records=records,original_median_us=statistics.median(a),
                candidate_median_us=statistics.median(b),both_candidate_blocks_faster=max(b)<min(a),
                latency_reduction_pct=(1-statistics.median(b)/statistics.median(a))*100,
                protocol_sha256=sha(out/'protocol.json'))
    save(out/'summary.json',result)
    print(result)


if __name__=='__main__':
    main()
