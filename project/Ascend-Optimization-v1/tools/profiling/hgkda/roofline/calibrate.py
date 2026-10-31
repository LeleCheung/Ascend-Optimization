"""按预登记小集合采集 Roofline 校准原件；调用者持卡7锁。"""
import argparse,datetime,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path

CASES=[dict(id='copy-large-persistent',kind='copy',elements=134217728,block=16384,programs=40),
       dict(id='gemm-fp32',kind='gemm',shape=[512,512,512],tile=[128,128,64],dtype='float32'),
       dict(id='gemm-bf16',kind='gemm',shape=[512,512,512],tile=[128,128,64],dtype='bfloat16')]

def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    a.output.mkdir(parents=True,exist_ok=True);state=a.output/'state.json'
    if state.exists():raise ValueError('不重复执行已有校准')
    info=dict(status='running',started=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=[])
    save(state,info)
    for spec in CASES:
        row=dict(id=spec['id'],status='running');info['cases'].append(row);save(state,info)
        directory=a.output/spec['id'];directory.mkdir();save(directory/'spec.json',spec)
        command=['msprof','op','--application='+shlex.join([sys.executable,str(Path(__file__).with_name('driver.py')),'--spec',str(directory/'spec.json'),'--output',str(directory/'compiled')]),
                 '--output='+str(directory/'profile'),'--kernel-name=roof_calibration_'+spec['kind'],
                 '--launch-count=1','--warm-up=10','--aic-metrics=Roofline']
        save(directory/'command.json',dict(command=command))
        try:
            with (directory/'stdout.txt').open('w') as out,(directory/'stderr.txt').open('w') as err:
                r=subprocess.run(command,stdout=out,stderr=err,timeout=600)
            tables=list((directory/'profile').rglob('OpBasicInfo.csv'))
            complete='Op profiling finish.' in (directory/'stdout.txt').read_text(errors='replace')
            row.update(status='completed' if r.returncode==0 and len(tables)==1 and complete else 'failed',returncode=r.returncode,tables=len(tables))
        except Exception as e:
            row.update(status='failed',error=str(e));save(state,info);raise
        save(state,info)
        if row['status']!='completed':raise RuntimeError('采集失败，保留原件并停止：'+spec['id'])
    info.update(status='completed',finished=datetime.datetime.now(datetime.timezone.utc).isoformat());save(state,info)

if __name__=='__main__':main()
