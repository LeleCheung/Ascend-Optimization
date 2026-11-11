"""独立核验累计积分的数学参考合同；保留原NPU默认合同及其失败记录。"""
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
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['repository','archive-project','original-project','regression-run','preceding-run',
                 'run-directory','lock','sdk']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--container',required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--device-python',required=True)
    parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args()
    run=args.run_directory.resolve(); run.mkdir(parents=True,exist_ok=False)
    receipt=run/'controller-state.json'
    state=dict(status='waiting_for_preceding_stages',contract_lane='qualified_mathematical_reference')
    save(receipt,state)

    def execute(label,command):
        with (run/(label+'.log')).open('xb') as output:
            result=subprocess.run(list(map(str,command)),cwd=args.repository,stdout=output,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(label+' 返回 '+str(result.returncode))

    def device(label,command):
        environment=dict(ASCEND_RT_VISIBLE_DEVICES=str(args.device),PYTHONDONTWRITEBYTECODE='1',
                         OMP_NUM_THREADS='1',MAX_JOBS='1',TRITON_CACHE_DIR=str(run/'triton-cache'),
                         PYTHONPATH=str(args.repository/'third_party/FlagGems/src'))
        shell='source '+shlex.quote(str(args.sdk/'set_env.sh'))+'\n'
        shell+='\n'.join('export '+k+'='+shlex.quote(v) for k,v in environment.items())+'\n'
        shell+='exec '+shlex.join([args.device_python,*map(str,command)])
        execute(label,['docker','exec',args.container,'bash','-c',shell])

    try:
        while True:
            audit=args.regression_run/'audit-final.json'
            previous=args.preceding_run/'controller-state.json'
            if previous.exists() and read(previous)['status'] in ['completed','requires_attention']:
                assert read(audit)['evidence_verified'] and read(audit)['verified']==25
                if (args.preceding_run/'state.json').exists():
                    assert read(args.preceding_run/'state.json')['status']=='completed'
                    assert (args.preceding_run/'service-stop.json').exists()
                break
            time.sleep(10)
        index=read(args.repository/'benchmarks/index.json')
        row=next(item for item in index['rows'] if item['operator']=='cumulative_trapezoid')
        original=read(args.repository/row['evidence']['supplementary']['artifact']['path'])

        def archived(path):
            return args.archive_project/Path(path).relative_to(args.original_project)

        leaf=archived(original['reproduce_command'][original['reproduce_command'].index('--leaf')+1])
        root=run/'operators/cumulative_trapezoid'; root.mkdir(parents=True)
        math_catalog=root/'supplementary-catalog'
        shutil.copytree(leaf,math_catalog,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        oracle=math_catalog/'oracle.py'
        binding='\n\n# 数学参考仅用于语义核验，不作为性能计时参考。\ndef run(y,*,dx=1,dim=-1):\n    return torch.cumulative_trapezoid(y.cpu(),dx=dx,dim=dim).to(y.device)\n'
        oracle.write_bytes(oracle.read_bytes()+binding.encode('utf8'))
        checker=root/'supplementary-checker.py'
        original_checker=archived(original['checker_source']['path'])
        assert sha(original_checker)==original['checker_source']['sha256']
        text=original_checker.read_bytes().decode('utf8')
        old="target_reference='当前冻结NPU参考；CPU PyTorch仅作旁列观察'"
        assert text.count(old)==1
        text=text.replace(old,"target_reference='独立数学合同：CPU PyTorch或经其核验的CANN组合；原NPU默认合同另存'")
        checker.write_bytes(text.encode('utf8'))
        save(run/'contract-change.json',dict(
            original_leaf=str(leaf),original_checker_sha256=sha(original_checker),checker_sha256=sha(checker),
            changed=['reference implementation','reference label'],
            preserved=['definition','correctness inputs','timing inputs','tolerances','timing settings',
                       'layout checks','fresh input checks','stream checks'],
            policy='独立数学参考合同；结果不冒充原NPU默认合同复现，不混入25项固定回归',
            pytorch_source='https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/Integration.cpp',
            accumulation_source='https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp'))
        with args.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state['status']='building_native_reference'; save(receipt,state)
            build=run/'native-build'
            device('build-native',[args.repository/'tools/backends/build_trapezoid.py','--sdk',args.sdk,
                                   '--output',build,'--parent-source',args.repository/row['source']['path']])
            built=read(build/'build.json')
            assert built['parent_source_sha256']==row['source']['sha256']
            state['status']='qualifying_mathematical_reference'; save(receipt,state)
            device('qualify-reference',[checker,'--operator','cumulative_trapezoid','--leaf',math_catalog,
                   '--source',built['reference'],'--output',run/'reference-qualification.json'])
            qualification=read(run/'reference-qualification.json')
            assert supplement_status('cumulative_trapezoid',qualification,built['reference_sha256'])=='passed'
            assert qualification['passed']==qualification['total']==original['total']
            source=root/'candidate.py'; shutil.copyfile(built['candidate'],source)
            review=dict(operator='cumulative_trapezoid',source=str(source),source_sha256=sha(source),
                        status='reviewed_strict_triton',all_branches_reviewed=True,dependencies_reviewed=True,
                        parent_source_sha256=row['source']['sha256'],parent_review_sha256=sha(args.repository/row['review']['path']),
                        method='保留加法与缩放的输入类型舍入，使用FP32扫描累加；全部布局路径仍为Triton',
                        review_notes=['逐路径审查父版本及两处增量计算、累加类型修改；无新增外部设备计算依赖。'])
            save(root/'review.json',review); admit(source.read_bytes().decode('utf8'),review)
            device('candidate-mathematical-supplement',[checker,'--operator','cumulative_trapezoid',
                   '--leaf',math_catalog,'--source',source,'--output',root/'supplementary.json'])
            assert supplement_status('cumulative_trapezoid',read(root/'supplementary.json'),sha(source))=='passed'
            device('capture-compiled',[args.repository/'tools/regression/capture_compiled.py','--source',source,
                   '--leaf',math_catalog,'--output',run/'compiled-path'])
            cann=root/'cann_supplementary-catalog'; shutil.copytree(math_catalog,cann)
            native=cann/'assets/native'; native.mkdir(parents=True)
            for path in [Path(built['reference']),Path(built['binary']),Path(built['source']),build/'build.json']:
                shutil.copyfile(path,native/path.name)
            text=(leaf/'oracle.py').read_bytes().decode('utf8')
            binding="""
import importlib.util as _native_import
from pathlib import Path as _NativePath
_native_spec=_native_import.spec_from_file_location('_cann_scan_reference',_NativePath(__file__).parent/'assets/native/reference.py')
_native_reference=_native_import.module_from_spec(_native_spec)
_native_spec.loader.exec_module(_native_reference)
def run(y,*,dx=1,dim=-1):
    return _native_reference.run(y,dx=dx,dim=dim)
"""
            (cann/'oracle.py').write_bytes((text+binding).encode('utf8'))
            for name in ['definition.json','correctness.jsonl','timing.jsonl']:
                assert (leaf/name).read_bytes()==(math_catalog/name).read_bytes()==(cann/name).read_bytes()
            old_result=args.archive_project/Path(row['evidence']['pytorch']['artifact']['original_path']).relative_to('project/Ascend-Optimization-v1')
            request=read(old_result.with_name(old_result.name.replace('.result.json','.request.json')))
            raw,digest=pack(cann); (root/'operator.tar').write_bytes(raw)
            request['binding']['bundle_id']='sha256:'+digest
            request['implementation']['sources'][0]['content']=source.read_bytes().decode('utf8')
            request['implementation']['name']+='_fp32_math_contract_20261109'
            save(root/'request.json',request)
            checks={}
            for kind,folder in [('supplementary',math_catalog),('cann_supplementary',cann)]:
                checks[kind]=dict(checker=str(checker),checker_sha256=sha(checker),leaf=str(folder),
                                 previous_total=original['total'],previous_passed=original['passed'],
                                 contract_sha256={name:sha(folder/name) for name in
                                                 ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']})
            job=dict(operator='cumulative_trapezoid',group=row['group'],order=row['order'],source=str(source),
                     source_sha256=sha(source),review=str(root/'review.json'),request=str(root/'request.json'),
                     archive=str(root/'operator.tar'),bundle_id='sha256:'+digest,checks=checks,
                     previous_cann_score=None,reference_kind=built['reference_kind'],settings=request['settings'])
            save(run/'plan.json',dict(schema_version='hgkda.regression/v1',repository=str(args.repository),
                 index_sha256=sha(args.repository/'benchmarks/index.json'),expected=1,jobs=[job],errors=[],
                 compatibility_files=[],policy=read(run/'contract-change.json'),
                 reference_qualification=dict(path=str(run/'reference-qualification.json'),sha256=sha(run/'reference-qualification.json'))))
        state['status']='formal_evaluation'; save(receipt,state)
        execute('formal-controller',[sys.executable,args.repository/'tools/regression/run.py','--plan',run/'plan.json',
                 '--container',args.container,'--device',args.device,'--port',args.port,'--lock',args.lock,
                 '--device-python',args.device_python,'--cann-environment',args.sdk/'set_env.sh'])
        for tool in ['audit','summarize']:
            execute(tool,[sys.executable,args.repository/('tools/regression/'+tool+'.py'),
                          '--run-directory',run,'--output',run/(tool+'-final.json')])
        state.update(status='completed',result=read(run/'state.json')); save(receipt,state)
    except Exception as error:
        state.update(status='requires_attention',error=type(error).__name__+': '+str(error)); save(receipt,state)
        raise


if __name__=='__main__':
    main()
