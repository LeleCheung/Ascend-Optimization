"""矩阵父版/PyTorch同输入profile；对多kernel参考明确登记所选组件。"""
import argparse,hashlib,json,re,shlex,subprocess,sys
from pathlib import Path

def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sides',nargs='+',default=['parent','pytorch']);p.add_argument('--sizes',nargs='+',type=int,default=[1024,2048]);p.add_argument('--dtypes',nargs='+',default=['float16','float32'])
    a=p.parse_args();a.output.mkdir(parents=True);state=dict(status='running',cases=[]);save(a.output/'state.json',state)
    for size in a.sizes:
      for dt in a.dtypes:
       for side in a.sides:
        directory=a.output/(side+'-'+dt+'-'+str(size));directory.mkdir();row=dict(case=directory.name,status='running');state['cases'].append(row);save(a.output/'state.json',state)
        spec=dict(side=side,shape=[size]*3,dtype=dt,source=str(a.source.resolve()),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest())
        save(directory/'spec.json',spec)
        app=shlex.join([sys.executable,str(Path(__file__).with_name('matrix_driver.py')),'--spec',str(directory/'spec.json'),'--output',str(directory/'compiled')])
        discovery=['msprof','op','--application='+app,'--kernel-name=__hgkda_roof_no_match__','--output='+str(directory/'discovery')]
        r=subprocess.run(discovery,capture_output=True,text=True,timeout=300);log=r.stdout+'\n'+r.stderr;(directory/'discovery.log').write_text(log)
        names=list(dict.fromkeys(re.findall(r'Kernel (.+?) skipped: not selected via --kernel-name\.',log)))
        if side=='pytorch':selected=[name for name in names if any(t in name.lower() for t in ['matmul','gemm'])]
        else:
            driver=json.loads((directory/'compiled/driver.json').read_text());symbols=[r['symbol'] for r in driver['compiled']]
            selected=[name for name in names if any(name==s or name.startswith(s+'_') for s in symbols)]
        save(directory/'discovery.json',dict(command=discovery,returncode=r.returncode,kernels=names,selected=selected))
        if r.returncode or len(selected)!=1:raise RuntimeError('kernel不唯一：'+str(selected))
        command=['msprof','op','--application='+app,'--kernel-name='+selected[0],'--output='+str(directory/'profile'),'--warm-up=10','--launch-count=1','--aic-metrics=Roofline']
        save(directory/'command.json',dict(command=command,scope='PyTorch只采GEMM组件，完整算子速度另用KGS；候选为已确认的融合kernel'))
        with (directory/'stdout.txt').open('w') as stdout,(directory/'stderr.txt').open('w') as stderr:r=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=600)
        complete='Op profiling finish.' in (directory/'stdout.txt').read_text(errors='replace')
        row.update(status='completed' if r.returncode==0 and complete else 'failed',returncode=r.returncode);save(a.output/'state.json',state)
        if row['status']!='completed':raise RuntimeError('采集失败：'+directory.name)
    state['status']='completed';save(a.output/'state.json',state)

if __name__=='__main__':main()
