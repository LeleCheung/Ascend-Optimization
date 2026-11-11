"""回归结束后核验 CANN 随机参考组合，并经独立 CLI 完成一次正式评测。"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from hgkda.adapters.bundle import pack
from hgkda.gate import admit
from hgkda.validation import supplement_status


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['repository','archive-project','original-project','regression-run','run-directory','lock','sdk']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--container',required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--device-python',required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--preceding-run',type=Path)
    parser.add_argument('--reuse-reference-run',type=Path)
    args=parser.parse_args()
    run=args.run_directory.resolve()
    run.mkdir(parents=True,exist_ok=False)
    state_path=run/'controller-state.json'
    state=dict(status='waiting_for_fixed_regression')
    save(state_path,state)
    try:
        while True:
            previous=read(args.regression_run/'state.json')
            if previous['status']=='requires_attention':
                raise RuntimeError('原正式评测任务未解决，不进入下一阶段')
            audit=args.regression_run/'audit-final.json'
            postflight=args.regression_run/'postflight-state.json'
            if audit.exists() and postflight.exists() and read(postflight)['status'] in ['completed','requires_attention']:
                assert read(audit)['evidence_verified'] and read(audit)['verified']==previous['expected']
                if args.preceding_run:
                    receipt=args.preceding_run/'controller-state.json'
                    if not receipt.exists() or read(receipt)['status'] not in ['completed','requires_attention']:
                        time.sleep(10)
                        continue
                    if (args.preceding_run/'state.json').exists():
                        assert read(args.preceding_run/'state.json')['status']=='completed'
                        assert (args.preceding_run/'service-stop.json').exists()
                break
            time.sleep(10)
        index=read(args.repository/'development/experiments/index.json')
        row=next(item for item in index['rows'] if item['operator']=='uniform')
        root=run/'operators/uniform'
        root.mkdir(parents=True)
        source=root/'candidate.py'
        source.write_bytes((args.repository/row['source']['path']).read_bytes())
        review=root/'review.json'
        shutil.copyfile(args.repository/row['review']['path'],review)
        assert sha(source)==row['source']['sha256']
        admit(source.read_bytes().decode('utf8'),read(review))
        prior=read(args.repository/row['evidence']['supplementary']['artifact']['path'])

        def archived(path):
            return args.archive_project/Path(path).relative_to(args.original_project)

        leaf=archived(prior['reproduce_command'][prior['reproduce_command'].index('--leaf')+1])
        catalog=root/'supplementary-catalog'
        shutil.copytree(leaf,catalog,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        checker=root/'supplementary-checker.py'
        checker.write_bytes(archived(prior['checker_source']['path']).read_bytes())
        assert sha(checker)==prior['checker_source']['sha256']
        random_checker=root/'random-checker.py'
        random_checker.write_bytes((args.archive_project/'tools/profiling/hgkda/check-endurance.py').read_bytes())
        assert sha(random_checker)==prior['random_details']['checker_sha256']
        # 保持冻结检查器的默认依赖布局，不改写其源码；正式控制器不需要额外私有参数。
        shutil.copyfile(random_checker,root.parent/'check-endurance.py')

        def execute(label,command):
            with (run/(label+'.log')).open('xb') as output:
                result=subprocess.run(list(map(str,command)),cwd=args.repository,stdout=output,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(label+' 返回 '+str(result.returncode))

        def device(label,command):
            environment=dict(ASCEND_RT_VISIBLE_DEVICES=str(args.device),PYTHONDONTWRITEBYTECODE='1',
                             OMP_NUM_THREADS='1',MAX_JOBS='1',TRITON_CACHE_DIR=str(run/'triton-cache'),
                             PYTHONPATH=str(args.repository/'development/dependencies/third_party/FlagGems/src'),
                             ASCEND_WORK_PATH=str(run/'vendor-work'),ASCEND_CACHE_PATH=str(run/'vendor-cache'))
            shell='source '+shlex.quote(str(args.sdk/'set_env.sh'))+'\n'
            shell+='\n'.join('export '+key+'='+shlex.quote(value) for key,value in environment.items())+'\n'
            shell+='exec '+shlex.join([args.device_python,*map(str,command)])
            execute(label,['docker','exec',args.container,'bash','-c',shell])

        with args.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state['status']='building_native_reference';save(state_path,state)
            build=run/'native-build'
            if args.reuse_reference_run:
                prior_build=args.reuse_reference_run/'native-build'
                built=read(prior_build/'build.json')
                assert sha(built['binary'])==built['binary_sha256']
                assert sha(built['reference'])==built['reference_sha256']
                assert sha(args.repository/'tools/operators/uniform/cann_uniform_range.cpp')==built['source_sha256']
                build.mkdir()
                for path in [Path(built['binary']),Path(built['reference']),Path(built['source']),prior_build/'build.json']:
                    shutil.copyfile(path,build/path.name)
            else:
                device('build-native',[args.repository/'tools/operators/uniform/build_uniform_range.py',
                                      '--sdk',args.sdk,'--output',build])
                built=read(build/'build.json')
            state['status']='qualifying_reference';save(state_path,state)
            if args.reuse_reference_run:
                old=args.reuse_reference_run
                assert not (old/'operators/uniform/evaluation').exists(), '有过正式提交时必须先查询原任务'
                for name in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']:
                    assert (catalog/name).read_bytes()==(old/'operators/uniform/supplementary-catalog'/name).read_bytes()
                assert sha(checker)==sha(old/'operators/uniform/supplementary-checker.py')
                assert sha(random_checker)==sha(old/'operators/uniform/random-checker.py')
                for name in ['reference-qualification.json','reference-qualification-random-details.json']:
                    shutil.copyfile(old/name,run/name)
                for name in ['supplementary.json','supplementary-random-details.json']:
                    shutil.copyfile(old/'operators/uniform'/name,root/name)
                save(run/'qualification-reuse.json',dict(original_run=str(old),
                     reference_qualification_sha256=sha(run/'reference-qualification.json'),
                     reason='上批仅合同上传目录不合规；未提交正式评测；二进制、源码、输入及检查器身份相同'))
            else:
                device('qualify-reference',[checker,'--operator','uniform','--leaf',catalog,
                      '--source',built['reference'],'--uniform-checker',random_checker,
                      '--output',run/'reference-qualification.json'])
            qualification=read(run/'reference-qualification.json')
            assert qualification['checker_source']['sha256']==sha(checker)
            assert supplement_status('uniform',qualification,built['reference_sha256'])=='passed'
            assert qualification['passed']==qualification['total']==prior['total']
            if not (root/'supplementary.json').exists():
                device('candidate-original-supplement',[checker,'--operator','uniform','--leaf',catalog,
                      '--source',source,'--uniform-checker',random_checker,'--output',root/'supplementary.json'])
            assert supplement_status('uniform',read(root/'supplementary.json'),sha(source))=='passed'
            cann=root/'cann_supplementary-catalog'
            shutil.copytree(catalog,cann)
            native=cann/'assets/native';native.mkdir(parents=True)
            for path in [Path(built['binary']),Path(built['reference']),Path(built['source']),build/'build.json']:
                shutil.copyfile(path,native/path.name)
            oracle=cann/'oracle.py'
            import ast
            text=oracle.read_text()
            function=next(node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name=='run')
            lines=text.splitlines(keepends=True)
            binding='''import importlib.util as _hgkda_import
from pathlib import Path as _HGKDAPath
_hgkda_spec=_hgkda_import.spec_from_file_location('_cann_uniform_reference',_HGKDAPath(__file__).parent/'assets/native/reference.py')
_hgkda_native=_hgkda_import.module_from_spec(_hgkda_spec)
_hgkda_spec.loader.exec_module(_hgkda_native)
def run(self,from_=0.,to=1.,*,generator=None):
    return _hgkda_native.run(self,from_,to,generator=generator)
'''
            oracle.write_text(''.join(lines[:function.lineno-1])+binding+''.join(lines[function.end_lineno:]),encoding='utf8')
            for name in ['definition.json','correctness.jsonl','timing.jsonl']:
                assert (catalog/name).read_bytes()==(cann/name).read_bytes()
            result_original=args.archive_project/Path(row['evidence']['pytorch']['artifact']['original_path']).relative_to('project/Ascend-Optimization-v1')
            request=read(result_original.with_name(result_original.name.replace('.result.json','.request.json')))
            assert request['implementation']['sources'][0]['content']==source.read_bytes().decode('utf8')
            raw,digest=pack(cann)
            (root/'operator.tar').write_bytes(raw)
            request['binding']['bundle_id']='sha256:'+digest
            save(root/'request.json',request)
            checks={}
            for kind,folder in [('supplementary',catalog),('cann_supplementary',cann)]:
                # 原检查器把扩展随机检查的默认路径作为启动参数；冻结为当前批次的同字节依赖。
                checks[kind]=dict(checker=str(checker),checker_sha256=sha(checker),leaf=str(folder),
                     previous_total=prior['total'],previous_passed=prior['passed'],
                     contract_sha256={name:sha(folder/name) for name in
                                      ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']})
            job=dict(operator='uniform',group=row['group'],order=row['order'],source=str(source),
                     source_sha256=sha(source),review=str(review),request=str(root/'request.json'),
                     archive=str(root/'operator.tar'),bundle_id='sha256:'+digest,checks=checks,
                     previous_cann_score=None,reference_kind=built['reference_kind'],settings=request['settings'])
            save(run/'plan.json',dict(schema_version='hgkda.regression/v1',repository=str(args.repository),
                 index_sha256=sha(args.repository/'development/experiments/index.json'),expected=1,jobs=[job],errors=[],compatibility_files=[],
                 reference_qualification=dict(path=str(run/'reference-qualification.json'),sha256=sha(run/'reference-qualification.json')),
                 policy='原严格Triton候选和测试合同；新增原生Uniform+Clamp参考；所有必要适配开销计入完整调用'))
        state['status']='formal_evaluation';save(state_path,state)
        execute('formal-controller',[sys.executable,args.repository/'tools/regression/run.py',
               '--plan',run/'plan.json','--container',args.container,'--device',args.device,'--port',args.port,
               '--lock',args.lock,'--device-python',args.device_python,'--cann-environment',args.sdk/'set_env.sh'])
        state.update(status='completed',result=read(run/'state.json'))
        save(state_path,state)
    except Exception as error:
        state.update(status='requires_attention',error=type(error).__name__+': '+str(error))
        save(state_path,state)
        raise


if __name__=='__main__':
    main()
