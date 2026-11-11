"""保留父版本合同，串行验证hardswish_out的较大连续分块候选。"""
import argparse
import ast
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
from hgkda.validation import supplement_status


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    temporary.replace(path)


def complete_block_loop(text):
    """整除时仅保留完整块循环，避免完整块与尾块同时占用片上缓冲。"""
    function = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == '_hardswish_stream')
    old = ast.get_source_segment(text, function)
    new = '''def _hardswish_stream(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, N, step):
        offset = start + lane
        if N % BLOCK == 0:
            x = tl.load(X + offset).to(tl.float32)
            inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
            tl.store(Y + offset, x * inner * (1.0 / 6.0))
        else:
            valid = offset < N
            x = tl.load(X + offset, valid, other=0).to(tl.float32)
            inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
            tl.store(Y + offset, x * inner * (1.0 / 6.0), valid)'''
    assert text.count(old) == 1
    return text.replace(old, new)


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
    parser.add_argument('--complete-block-loop',action='store_true')
    parser.add_argument('--affine-gate',action='store_true')
    parser.add_argument('--native-low-precision',action='store_true')
    parser.add_argument('--half-only-large-tiles',action='store_true')
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
        if args.half_only_large_tiles:
            assert args.dtype_guard
            old_cap = '4096 if y.dtype == torch.float32 or out.dtype == torch.float32 else 8192'
            assert text.count(old_cap) == 1
            text = text.replace(old_cap, '8192 if y.dtype == torch.float16 and out.dtype == torch.float16 else 4096')
        if args.disable_large_multibuffer:
            assert args.dtype_guard
            replacements={
                '_hardswish_stream.warmup(y, out, n, block, grid=(1,))':'_hardswish_stream.warmup(y, out, n, block, grid=(1,), multibuffer=block <= 4096)',
                '_hardswish_stream[plan[4],](y, out, n, plan[5])':'_hardswish_stream[plan[4],](y, out, n, plan[5], multibuffer=plan[5] <= 4096)',
                '_hardswish_stream[f[4],](x, out, n, f[5])':'_hardswish_stream[f[4],](x, out, n, f[5], multibuffer=f[5] <= 4096)'}
            for before,after in replacements.items():
                assert text.count(before)==1
                text=text.replace(before,after)
        if args.complete_block_loop:
            text = complete_block_loop(text)
        if args.affine_gate:
            assert args.complete_block_loop
            old_inner = 'inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)'
            assert text.count(old_inner) == 2 and text.count('x * inner * (1.0 / 6.0)') == 2
            text = text.replace(old_inner, 'inner = tl.minimum(tl.maximum(x * (1.0 / 6.0) + 0.5, 0.0), 1.0)')
            text = text.replace('x * inner * (1.0 / 6.0)', 'x * inner')
        if args.native_low_precision:
            assert args.affine_gate
            function = next(node for node in ast.parse(text).body
                            if isinstance(node, ast.FunctionDef) and node.name == '_hardswish_stream')
            old_function = ast.get_source_segment(text, function)
            assert old_function.count('.to(tl.float32)') == 2
            new_function = old_function.replace('.to(tl.float32)',
                '.to(tl.float32 if X.dtype.element_ty == tl.float32 or Y.dtype.element_ty == tl.float32 else X.dtype.element_ty)')
            text = text.replace(old_function, new_function)
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
        if args.complete_block_loop:
            review['method'] += '；整除时仅编译完整块循环，非整除保留掩码；不同时分配完整块与尾块缓冲'
        if args.affine_gate:
            review['method'] = review['method'].replace('设备数学及回退不变', '回退不变')
            review['method'] += '；归一化门控因子clamp(x/6+1/2,0,1)，减少一次中间向量乘法；原容差和特殊值检查保持不变'
        if args.native_low_precision:
            review['method'] += '；输入或输出为FP32时保留FP32计算，其余采用输入精度减少片上中间缓冲；先通过冻结补测再提交正式评测'
        if args.half_only_large_tiles:
            review['method'] += '；依据实际编译结果，仅FP16输入与输出使用8192分块，BF16和FP32保留4096，避免BF16内部扩展造成UB溢出'
        for bridge in review['reviewed_host_bridges']:bridge['candidate_sha256']=sha(source)
        save(root/'review.json',review);admit(text,review)
        request=read(root/'request.json')
        request['implementation']['sources'][0]['content']=text
        request['implementation']['name']+='_tile8192_guarded_20261109' if args.dtype_guard else '_tile8192_20261109'
        if args.complete_block_loop:
            request['implementation']['name'] += '_complete_block_loop'
        if args.affine_gate:
            request['implementation']['name'] += '_affine_gate'
        if args.native_low_precision:
            request['implementation']['name'] += '_native_low_precision'
        if args.half_only_large_tiles:
            request['implementation']['name'] += '_half_only_large_tiles'
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
            command=[args.device_python,checks['supplementary']['checker'],'--operator','hardswish_out',
                     '--source',str(source),'--leaf',checks['supplementary']['leaf'],
                     '--output',str(run/'candidate-qualification.json')]
            shell='source '+shlex.quote(str(args.sdk/'set_env.sh'))+'\n'
            shell+='\n'.join('export '+k+'='+shlex.quote(v) for k,v in environment.items())+'\nexec '+shlex.join(command)
            execute('candidate-qualification',['docker','exec',args.container,'bash','-c',shell])
            value=read(run/'candidate-qualification.json')
            assert supplement_status('hardswish_out',value,sha(source))=='passed'
            assert value['passed']==value['total']==checks['supplementary']['previous_total']
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
