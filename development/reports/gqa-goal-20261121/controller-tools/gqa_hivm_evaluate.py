"""Validate bound HIVM artifacts through the original GQA four-case contract."""
import argparse
import fcntl
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gqa-run', type=Path, required=True)
    parser.add_argument('--ir-run', type=Path, required=True)
    parser.add_argument('--parent-invocation', required=True)
    parser.add_argument('--compile-result', type=Path, required=True)
    parser.add_argument('--reference-binary', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    root, run = args.gqa_run.resolve(), args.ir_run.resolve()
    tools = Path(__file__).resolve().parent
    parent = root/'iterations'/args.parent_invocation
    config = read(root/'config.json')
    source = run/'workspace/frozen.py'
    assert sha(source) == sha(parent/'workspace/candidate.py')
    compiled = read(args.compile_result)
    assert compiled['status'] == 'compiled'
    assert compiled['reference_binary_sha256'] == sha(args.reference_binary)
    if args.baseline:
        assert compiled['execution_matches_reference']
    else:
        review = read(run/'agent-calls'/args.name/'host-review.json')
        assert review['status'] == 'semantics_and_cpu_approved'
        assert review['changed_ir_sha256'] == compiled['ir_sha256']
    val = run/('validation-'+args.name)
    val.mkdir(parents=True, exist_ok=False)
    original = parent/'capture-parent/candidate/actual-ir/_gqa_fwd_kernel.bcmlir'
    artifact = dict(stage='hivm', binary=compiled['binary'], binary_sha256=compiled['binary_sha256'],
                    reference_binary=str(args.reference_binary.resolve()), reference_binary_sha256=sha(args.reference_binary),
                    compile_receipt=str(args.compile_result.resolve()), compile_receipt_sha256=sha(args.compile_result))
    manifest = dict(source=str(source), source_sha256=sha(source), contract=str(root/'contract.json'),
        contract_sha256=sha(root/'contract.json'), original_ir=str(original), original_ir_sha256=sha(original),
        changed_ir=compiled['ir'], changed_ir_sha256=compiled['ir_sha256'],
        original_source_location=str(parent/'workspace/candidate.py'), symbol='_gqa_fwd_kernel',
        target_marker='strides: [8192, 1]', target_aliases={'Ascend910B4': 'Ascend910B4-1'},
        cache=str(val/'cache'), receipts=str(val/'compiler-receipts.jsonl'), compiled_artifact=artifact,
        scope='Fixed-source HIVM scheduling. Exact target BCMLIR and original executable validated before artifact insertion; other specializations compile normally.')
    save(val/'manifest.json', manifest)
    command = [config['device_python'], str(tools/'gqa_ir_bench_driver.py'), '--manifest', str(val/'manifest.json'),
               '--harness',str(root/'tools/attention_bench.py'),'--contract',manifest['contract'],
               '--candidate',manifest['source'],'--output',str(val/'results.json')]
    shell = 'source '+shlex.quote(config['sdk']+'/set_env.sh')+'\n'
    for key,value in dict(ASCEND_RT_VISIBLE_DEVICES=str(config['device']),OMP_NUM_THREADS='1',
                         TRITON_DISABLE_LINE_INFO='false',PYTHONDONTWRITEBYTECODE='1').items():
        shell += 'export '+key+'='+shlex.quote(value)+'\n'
    shell += 'exec '+shlex.join(command)
    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (val/'execution.log').open('x') as log:
            process = subprocess.run(['docker','exec',config['container'],'bash','-c',shell],
                                     stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    save(val/'execution.json',dict(command=command,returncode=process.returncode))
    assert process.returncode == 0
    result = read(val/'results.json')
    qualified = {row['case_id'] for row in result['cases']} == {1,2,4,7} and all(
        row['status']=='measured' and all(row[side+'_correctness']['official_checker']['passed']
        for side in ['candidate','native']) for row in result['cases'])
    target = next(row for row in result['cases'] if row['case_id']==4)
    review = dict(status='original_four_passed' if qualified else 'four_case_failure',
                  source_sha256=sha(source),ir_sha256=compiled['ir_sha256'],
                  target_us=target.get('candidate_timing',{}).get('full_call_us'),
                  baseline=args.baseline,post_status='original_binary_evidence_reused' if args.baseline else 'pending')
    save(val/'reviewed-result.json',review)
    if not qualified or args.baseline:
        print(review)
        return
    post = run/('post-'+args.name)
    (post/'workspace').mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source,post/'workspace/candidate.py')
    shutil.copyfile(root/'contract.json',post/'contract.json')
    shutil.copytree(root/'tools',post/'tools')
    for name in ['gqa_bcmlir_runtime.py','gqa_ir_bench_driver.py','gqa_binary_execution.py']:
        shutil.copyfile(tools/name,post/'tools'/name)
    post_manifest = dict(manifest,source=str(post/'workspace/candidate.py'),cache=str(post/'compiled-cache'),
                         receipts=str(post/'compiler-receipts.jsonl'))
    save(post/'bcmlir-manifest.json',post_manifest)
    config.update(run_directory=str(post),trace_round_id='hivm-'+args.name,
                  bcmlir_manifest=str(post/'bcmlir-manifest.json'),
                  existing_validation=dict(result=str(val/'results.json'),result_sha256=sha(val/'results.json'),
                                           cache=str(val/'cache')))
    save(post/'config.json',config)
    save(post/'retained-parent.json',dict(source_sha256=sha(source),status='fixed_source_hivm_post_pending'))
    command = [sys.executable,str(tools/'gqa_continue_observe.py'),'--run',str(post),'--simulator-minutes','3']
    with (post/'driver.log').open('x') as log:
        process = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    save(post/'driver-execution.json',dict(command=command,returncode=process.returncode))
    assert process.returncode == 0
    from gqa_binding_facts import facts
    binding = facts(post)
    save(post/'binary-binding-and-facts.json',binding)
    review.update(post_status='complete',binary_binding=str(post/'binary-binding-and-facts.json'))
    save(val/'reviewed-result.json',review)
    print(review)


if __name__ == '__main__':
    main()
