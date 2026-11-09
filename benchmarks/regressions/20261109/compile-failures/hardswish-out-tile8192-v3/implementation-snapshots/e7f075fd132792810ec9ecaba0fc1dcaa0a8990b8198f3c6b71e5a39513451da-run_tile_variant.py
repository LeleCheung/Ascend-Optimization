"""保留父版本合同，串行验证hardswish_out的较大连续分块候选。"""
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

from hgkda.gate import admit


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['repository','regression-run','preceding-run','run-directory','lock','sdk']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--container',required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--device-python',required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--dtype-guard',action='store_true')
    parser.add_argument('--disable-large-multibuffer',action='store_true')
    args=parser.parse_args()
    run=args.run_directory.resolve();run.mkdir(parents=True,exist_ok=False)
    receipt=run/'controller-state.json';state=dict(status='waiting_for_preceding_stages');save(receipt,state)

    def execute(label,command):
        with (run/(label+'.log')).open('xb') as output:
            result=subprocess.run(list(map(str,command)),cwd=args.repository,stdout=output,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(label+' 返回 '+str(result.returncode))

    try:
        while True:
            previous=args.preceding_run/'controller-state.json'
            if previous.exists() and read(previous)['status'] in ['completed','requires_attention']:
                audit=read(args.regression_run/'audit-final.json')
                assert audit['evidence_verified'] and audit['verified']==25
                if (args.preceding_run/'state.json').exists():
                    assert read(args.preceding_run/'state.json')['status']=='completed'
                    assert (args.preceding_run/'service-stop.json').exists()
                break
            time.sleep(10)
        original=read(args.regression_run/'plan.json')
        job=next(item for item in original['jobs'] if item['operator']=='hardswish_out')
        root=run/'operators/hardswish_out';root.mkdir(parents=True)
        for name in ['request.json','review.json','operator.tar']:
            shutil.copyfile(args.regression_run/'operators/hardswish_out'/name,root/name)
        checks={}
        for kind,spec in job['checks'].items():
            checker=root/(kind+'-checker.py');shutil.copyfile(spec['checker'],checker)
            leaf=root/(kind+'-catalog');shutil.copytree(spec['leaf'],leaf,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            checks[kind]=dict(spec,checker=str(checker),leaf=str(leaf))
        text=Path(job['source']).read_bytes().decode('utf8')
        old='block = min(4096, max(32, 1 << (n - 1).bit_length()))'
        assert text.count(old)==1
        if args.dtype_guard:
            text=text.replace(old,'block = min(cap, max(32, 1 << (n - 1).bit_length()))')
            assert text.count('def _geometry(n, device):')==1
            text=text.replace('def _geometry(n, device):','def _geometry(n, device, cap):')
            assert text.count('_geometry(n, device)')==1
            text=text.replace('_geometry(n, device)','_geometry(n, device, 4096 if y.dtype == torch.float32 or out.dtype == torch.float32 else 8192)')
        else:
            text=text.replace(old,'block = min(8192, max(32, 1 << (n - 1).bit_length()))')
        if args.disable_large_multibuffer:
            assert args.dtype_guard
            replacements={
                '_hardswish_stream.warmup(y, out, n, block, grid=(1,))':'_hardswish_stream.warmup(y, out, n, block, grid=(1,), multibuffer=block <= 4096)',
                '_hardswish_stream[plan[4],](y, out, n, plan[5])':'_hardswish_stream[plan[4],](y, out, n, plan[5], multibuffer=plan[5] <= 4096)',
                '_hardswish_stream[f[4],](x, out, n, f[5])':'_hardswish_stream[f[4],](x, out, n, f[5], multibuffer=f[5] <= 4096)'}
            for before,after in replacements.items():
                assert text.count(before)==1
                text=text.replace(before,after)
        review=read(root/'review.json')
        for bridge in review['reviewed_host_bridges']:
            record=next(item for item in original['compatibility_files'] if item['path']==bridge['binary'])
            for field in ['binary','cpp_source','build_record']:
                archived=Path(record['archived']).with_name(Path(bridge[field]).name)
                assert sha(archived)==bridge[field+'_sha256']
                text=text.replace(bridge[field],str(archived))
                bridge[field]=str(archived)
        source=root/'candidate.py';source.write_bytes(text.encode('utf8'))
        review.update(source=str(source),source_sha256=sha(source),parent_source_sha256=job['source_sha256'],
                      method='连续分块上限4096增至8192；宿主桥重定位到同SHA归档；设备数学及所有回退不变')
        if args.dtype_guard:
            review['method']='低精度分块上限8192，FP32或FP32输出保留4096；大分块按参数关闭自动多缓冲；设备数学及回退不变'
        for bridge in review['reviewed_host_bridges']:bridge['candidate_sha256']=sha(source)
        save(root/'review.json',review);admit(text,review)
        request=read(root/'request.json')
        request['implementation']['sources'][0]['content']=text
        request['implementation']['name']+='_tile8192_guarded_20261109' if args.dtype_guard else '_tile8192_20261109'
        save(root/'request.json',request)
        job=dict(job,source=str(source),source_sha256=sha(source),review=str(root/'review.json'),
                 request=str(root/'request.json'),archive=str(root/'operator.tar'),checks=checks,
                 previous_cann_score=read(args.regression_run/'state.json')['operators']['hardswish_out']['evaluation']['geo_mean'])
        save(run/'plan.json',dict(original,jobs=[job],expected=1,policy='独立较大连续分块候选；原正式回归不修改'))
        with args.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state['status']='capturing_compiled_paths';save(receipt,state)
            environment=dict(ASCEND_RT_VISIBLE_DEVICES=str(args.device),PYTHONDONTWRITEBYTECODE='1',
                             OMP_NUM_THREADS='1',TRITON_CACHE_DIR=str(run/'triton-cache'),
                             PYTHONPATH=str(args.repository/'third_party/FlagGems/src'))
            command=[args.device_python,str(args.repository/'tools/regression/capture_compiled.py'),
                     '--source',str(source),'--leaf',checks['supplementary']['leaf'],'--output',str(run/'compiled-path')]
            shell='source '+shlex.quote(str(args.sdk/'set_env.sh'))+'\n'
            shell+='\n'.join('export '+k+'='+shlex.quote(v) for k,v in environment.items())+'\nexec '+shlex.join(command)
            execute('capture-compiled',['docker','exec',args.container,'bash','-c',shell])
        state['status']='formal_evaluation';save(receipt,state)
        execute('formal-controller',[sys.executable,args.repository/'tools/regression/run.py','--plan',run/'plan.json',
                 '--container',args.container,'--device',args.device,'--port',args.port,'--lock',args.lock,
                 '--device-python',args.device_python,'--cann-environment',args.sdk/'set_env.sh'])
        for tool in ['audit','summarize']:
            execute(tool,[sys.executable,args.repository/('tools/regression/'+tool+'.py'),
                          '--run-directory',run,'--output',run/(tool+'-final.json')])
        state.update(status='completed',result=read(run/'state.json'));save(receipt,state)
    except Exception as error:
        state.update(status='requires_attention',error=type(error).__name__+': '+str(error));save(receipt,state)
        raise


if __name__=='__main__':main()
