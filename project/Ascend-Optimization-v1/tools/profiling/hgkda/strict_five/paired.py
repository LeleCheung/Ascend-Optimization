"""独立msprof op成对采集；固定原PyTorch oracle，不改共享KGS或采集成功条件。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys

HERE=Path(__file__).resolve().parent
DRIVER=HERE.parent/'next_phase/driver.py'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def run(command,log,timeout=300):
    with log.open('wb') as stream:
        process=subprocess.Popen([str(x) for x in command],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # 仅清理本次新建的进程组；先等其退出再允许后续设备工作。
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            raise RuntimeError('独立采集超时，已终止本采集进程组：'+str(log))
    if code:raise RuntimeError('独立采集失败：'+str(log))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    assert not args.output.exists(),'采集目录已有原件，不覆盖重跑'
    args.output.mkdir(parents=True)
    spec=json.loads(args.spec.read_text());leaf=Path(spec['leaf']);source=Path(spec['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==spec['source_sha256']
    state=dict(schema='hgkda-strict-five-paired-v1',status='running',spec=spec,
               collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               driver_sha256=hashlib.sha256(DRIVER.read_bytes()).hexdigest(),
               oracle_sha256=hashlib.sha256((leaf/'oracle.py').read_bytes()).hexdigest(),sides={})
    (args.output/'spec.json').write_bytes(args.spec.read_bytes())
    save(args.output/'state.json',state)
    for name,driver_side in [('candidate','candidate'),('reference','pytorch')]:
        out=args.output/name;out.mkdir()
        row=dict(status='running',meaning='严格Triton候选' if name=='candidate' else '冻结原PyTorch oracle；原生参考资格仍单列')
        state['sides'][name]=row
        command=[sys.executable,DRIVER,'--spec',args.spec,'--side',driver_side]
        try:
            run(command+['--semantic',out/'semantics.json'],out/'semantic.log')
            run(command+['--trace',out/'trace.json'],out/'trace.log')
            trace=json.loads((out/'trace.json').read_text())
            events=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
            targets=list(dict.fromkeys(e['name'] for e in events if e.get('ph')=='X' and e.get('name') and
                ('kernel' in str(e.get('cat','')).lower() or 'Task Type' in e.get('args',{}))))
            assert targets,'目标调用范围内未发现可绑定的设备任务'
            application=shlex.join([str(x) for x in command])
            discovery=out/'discovery.log'
            run(['msprof','op','--application='+application,'--kernel-name=__hgkda_no_match__',
                 '--launch-count=1','--warm-up=0','--aic-metrics=Roofline','--output='+str(out/'discovery')],discovery,180)
            discovered=list(dict.fromkeys(re.findall(r'Kernel (.+?) skipped: not selected via --kernel-name\.',discovery.read_text(errors='replace'))))
            if set(targets).issubset(discovered):names=targets
            elif len(targets)==1 and len(discovered)==1:names=discovered
            else:raise RuntimeError('目标trace与实际符号无法唯一绑定，拒绝混入输入生成kernel：'+repr(dict(trace=targets,op=discovered)))
            row['kernel_name_binding']=dict(trace_names=targets,discovered=discovered,op_names=names,
                                           policy='精确符号对应，或唯一目标任务与唯一实际kernel对应')
            row['kernels']=names
            for index,symbol in enumerate(names):
                folder=out/'profile' if len(names)==1 else out/('kernel-%03d'%index)
                run(['msprof','op','--application='+application,'--kernel-name='+symbol,
                     '--launch-count=1','--warm-up=0','--aic-metrics=Roofline','--output='+str(folder)],out/('kernel-%03d.log'%index),480)
                basic=list(folder.rglob('OpBasicInfo.csv'));assert len(basic)==1
                with basic[0].open(encoding='utf-8-sig',newline='') as stream:info=list(csv.DictReader(stream))
                assert len(info)==1 and info[0]['Op Name']==symbol,'采集目标符号不一致'
            row['metrics_files']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*.csv')}
            assert row['metrics_files'],'缺少真实指标CSV'
            if len(names)==1:
                save(out/'collection.json',dict(status='completed',side=name,
                     spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),groups=['Roofline'],
                     expected_kernel=names[0],reference_kind='original_PyTorch_only' if name=='reference' else None))
            if name=='candidate':run(command+['--ir',out/'ir'],out/'ir.log',480)
            row['status']='completed'
        except Exception as error:
            row.update(status='failed',error=type(error).__name__+': '+str(error))
        save(args.output/'state.json',state)
    state['status']='completed' if all(r['status']=='completed' for r in state['sides'].values()) else 'partial'
    save(args.output/'state.json',state)
    # 绝对指标保留，不把未配对或不同作用域的值强行相除。
    summary=dict(schema=state['schema'],status=state['status'],spec=spec,
                 raw_state=str(args.output/'state.json'),sides=state['sides'],
                 scope='同一冻结输入的单次代表采集；真实完整调用性能以KGS为准。比例不代表物理利用率上限。')
    if state['status']=='completed' and all(len(v.get('kernels',[]))==1 for v in state['sides'].values()):
        compare=HERE.parent/'skills/hgkda/scripts/compare_opprof.py'
        try:
            run([sys.executable,compare,'--directory',args.output,'--output',args.output/'comparison.json'],args.output/'compare.log',60)
            comparison=json.loads((args.output/'comparison.json').read_text())
            summary['comparison']=dict(path=str(args.output/'comparison.json'),
                sha256=hashlib.sha256((args.output/'comparison.json').read_bytes()).hexdigest(),
                differences=[v for v in comparison['differences'] if re.search(
                    'mte|vector|scalar|wait|conflict|GM_to|UB_to|time|bandwidth|instructions',v['metric'],re.I)],
                derived_differences=comparison['derived_differences'],limitations=comparison['limitations'])
        except Exception as error:summary['comparison_error']=str(error)
    save(args.output/'summary.json',summary)
    print(json.dumps(dict(status=state['status'],sides={k:v['status'] for k,v in state['sides'].items()})))


if __name__=='__main__':main()
