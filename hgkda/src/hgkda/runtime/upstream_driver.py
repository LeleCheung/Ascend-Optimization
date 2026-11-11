"""按明确文件运行未修改的上游测试，复用上游结果解析函数。"""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_bytes());repo=Path(c['repository']).resolve()
    operator=c['operator'];folder=a.output/operator;folder.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(repo/'tools'))
    upstream=runpy.run_path(str(repo/'tools/run_tests.py'),run_name='hgkda_upstream_parser')
    upstream['probe_env']()
    results={}
    for phase,subdir,key in [('accuracy','tests','accuracy_files'),('performance','benchmark','benchmark_files')]:
        paths=[]
        for name in c[key]:
            path=(repo/name).resolve();assert path.is_relative_to(repo/subdir) and path.is_file()
            paths.append(str(path.relative_to(repo/subdir)))
        assert paths,'必须明确目标测试文件'
        target=folder/(phase+'_result.json')
        command=[sys.executable,'-m','pytest',*paths,'-m',c.get('marker',operator),
                 '--record','json','--output',str(target),'--continue-on-collection-errors']
        command+=['--ref',c.get('accuracy_reference','cpu'),'-vs'] if phase=='accuracy' else ['--level','core']
        (folder/(phase+'_command.json')).write_text(json.dumps(command,indent=2),encoding='utf8')
        with (folder/(phase+'_stdout.log')).open('xb') as out,(folder/(phase+'_stderr.log')).open('xb') as err:
            process=subprocess.run(command,cwd=repo/subdir,stdout=out,stderr=err,timeout=c.get('timeout_seconds',3600))
        if target.exists():
            value=upstream['parse_accuracy_data'](target) if phase=='accuracy' else upstream['parse_perf_data'](operator,target)
        else:value=dict(status='NotFound')
        value['exit_code']=process.returncode;value['data_file']=str(target.relative_to(a.output));results[phase]=value
    import flag_gems
    results['customized']=operator in [x[0] for x in flag_gems.runtime.backend.get_customized_ops()]
    value=dict(result={operator:results},env=upstream['ENV_INFO'],scope='upstream_operator_files_tests_and_core_benchmark')
    (a.output/'summary.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    return 0 if all(results[p]['exit_code']==0 for p in ['accuracy','performance']) else 1


if __name__=='__main__':raise SystemExit(main())
