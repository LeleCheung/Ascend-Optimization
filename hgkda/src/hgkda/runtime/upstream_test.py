"""Run native tests in an explicitly supplied external FlagGems worktree and preserve separate acceptance evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from hgkda.adapters.flaggems import summarize
from hgkda.gate import admit


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def save(path,value):
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


def snapshot(repo, expected, allowed):
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args]).decode('utf8').strip()
    assert git('rev-parse','HEAD')==expected,'Target repository commit changed'
    changed=set(filter(None,git('diff','--name-only','--no-renames','HEAD').splitlines()))
    untracked=set(filter(None,git('ls-files','--others','--exclude-standard').splitlines()))
    assert changed.union(untracked).issubset(allowed),'External worktree contains undeclared changes: '+str(changed.union(untracked)-set(allowed))
    for name,want in allowed.items():
        path=(repo/name).resolve()
        assert path.is_relative_to(repo) and name.startswith('src/flag_gems/') and path.is_file()
        assert sha(path)==want,'Integration-file identity changed: '+name
    return dict(commit=expected,changed=sorted(changed),untracked=sorted(untracked),files=allowed,
                runner_sha256=sha(repo/'tools/run_tests.py'),
                diff=subprocess.check_output(['git','-C',str(repo),'diff','--binary','HEAD']).decode('utf8'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args(argv);c=read(a.config)
    repo=Path(c['repository']).resolve();out=Path(c['output']).resolve()
    assert re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',c['operator'])
    assert not out.is_relative_to(repo),'Output must be outside the external worktree'
    before=snapshot(repo,c['upstream_commit'],c['integration_files'])
    source=Path(c['delivery_source']);assert sha(source)==c['delivery_sha256']
    assert c['expected_function_source'] in c['integration_files']
    review=read(c['source_review'])
    admit((repo/c['expected_function_source']).read_bytes().decode('utf8'),review)
    assert isinstance(c['device'],int) and c['device']>=0
    out.mkdir(parents=True,exist_ok=False)
    # Upstream uses pytest; pin its interpreter to avoid accidentally selecting another environment.
    bin_dir=out/'bin';bin_dir.mkdir()
    pytest_entry=bin_dir/'pytest'
    pytest_entry.write_text('#!'+c['python']+'\nimport pytest\nraise SystemExit(pytest.console_main())\n',encoding='utf8')
    pytest_entry.chmod(0o755)
    save(out/'identity.json',dict(repository=str(repo),delivery_source=str(source),delivery_sha256=sha(source),
         adapter_sha256=sha(__file__),source_review_sha256=sha(c['source_review']),
         started_at=datetime.now(timezone.utc).isoformat(),**before))
    save(out/'config.json',c)
    for name in c['integration_files']:
        target=out/'integration'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((repo/name).read_bytes())
    env=dict(c.get('environment',{}),PYTHONPATH=str(repo/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
             ASCEND_RT_VISIBLE_DEVICES=str(c['device']),NPU_VISIBLE_DEVICES=str(c['device']),
             TRITON_CACHE_DIR=str(out/'triton-cache'))
    def execute(label,arguments):
        shell='source '+shlex.quote(str(Path(c['sdk'])/'set_env.sh'))+'\n'
        shell+='\n'.join('export '+k+'='+shlex.quote(str(v)) for k,v in env.items())+'\n'
        shell+='export PATH='+shlex.quote(str(bin_dir)+':'+str(Path(c['python']).parent))+':"$PATH"\n'
        shell+='exec '+shlex.join(list(map(str,arguments)))
        command=['docker','exec','-w',str(repo),c['container'],'bash','-c',shell] if c.get('container') else ['bash','-c',shell]
        with (out/(label+'.log')).open('xb') as log:
            return subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=c.get('timeout_seconds',3600)).returncode
    import fcntl
    with Path(c['lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        code=execute('import-probe',[c['python'],Path(__file__).with_name('upstream_probe.py'),
             '--repository',repo,'--operator',c['operator'],'--output',out/'import.json'])
        assert code==0,'Import verification failed; see import-probe.log'
        imported=read(out/'import.json');assert imported['source']==c['expected_function_source']
        assert imported['source_sha256']==c['integration_files'][imported['source']]
        if c.get('require_kgs_entrypoint'):
            assert imported['kgs_same_entrypoint'] is True,'KGS run and operator export must bind to the same function object'
        mode=c.get('mode','scheduler')
        if mode=='operator-files':
            assert c.get('accuracy_files') and c.get('benchmark_files')
            command=[c['python'],Path(__file__).with_name('upstream_driver.py'),'--config',a.config.resolve(),'--output',out/'upstream']
        else:
            assert mode=='scheduler'
            command=[c['python'],repo/'tools/run_tests.py','--ops',c['operator'],'--gpus',str(c['device']),
                     '--dump-output','--output-dir',out/'upstream','--color','never']
        save(out/'command.json',dict(argv=list(map(str,command)),quick=False,mode=mode))
        code=execute('upstream',command)
    summary=out/'upstream/summary.json'
    result=summarize(read(summary) if summary.exists() else {},c['operator'],code)
    result['mode']=mode
    if mode=='operator-files':result['scope']='upstream_operator_files_tests_and_core_benchmark'
    if c.get('expected_accuracy_cases_from'):
        expected_cases=read(c['expected_accuracy_cases_from'])
        actual_cases=read(out/'upstream'/c['operator']/'accuracy_result.json')
        result['same_accuracy_case_set']=set(expected_cases)==set(actual_cases)
        assert result['same_accuracy_case_set'],'Explicit-file run lost or changed the originally selected cases'
    for name,want in c['integration_files'].items():assert sha(repo/name)==want
    # Upstream scripts may create logs; allow only declared source patches while keeping tests/configuration unchanged.
    after_diff=subprocess.check_output(['git','-C',str(repo),'diff','--binary','HEAD']).decode('utf8')
    assert after_diff==before['diff'],'Tracked source or tests changed during execution'
    result.update(returncode=code,upstream_commit=c['upstream_commit'],delivery_sha256=c['delivery_sha256'],
                  import_identity=imported,finished_at=datetime.now(timezone.utc).isoformat())
    save(out/'result.json',dict(schema_version='hgkda.result/v1',operation='upstream-test',status='completed',result=result))
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
