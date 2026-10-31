"""新采集同输入候选/PyTorch证据，复用已有采集器并核验身份。"""
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--side',choices=['seed','candidate'],default='seed');a=p.parse_args()
    b=a.batch.resolve();side=a.side
    source=b/('seed.py' if side=='seed' else 'candidate.py')
    if side=='seed' and not source.exists():
        source.write_bytes((PROJECT/'operators/campaigns/roofline-20261027/validation/delivered.py').read_bytes())
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    root=b/(side+'-profile');assert not root.exists()
    sides=['candidate','pytorch'] if side=='seed' else ['candidate']
    command=[sys.executable,str(HERE.parent/'roofline/collect_matrix.py'),'--source',str(source),'--output',str(root),'--sides',*sides,'--sizes','512','1024','--dtypes','float16']
    with (b/(side+'-collection.log')).open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=2400,check=True)
    if side=='seed':
        for size in [512,1024]:
            out=b/'paired'/('float16-'+str(size));out.mkdir(parents=True)
            spec=dict(operator='matmul_bias_activation',shape=[size]*3,dtype='float16',seed=20261027,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                reference_scope='PyTorch GEMM组件；候选为融合bias+ReLU的kernel，组件指标不作为完整算子加速比')
            save(out/'spec.json',spec);digest=hashlib.sha256((out/'spec.json').read_bytes()).hexdigest()
            for label,original in [('reference','pytorch'),('candidate','candidate')]:
                directory=root/(original+'-float16-'+str(size));discovery=json.loads((directory/'discovery.json').read_bytes())
                target=out/label;target.mkdir();(target/'profile').symlink_to(directory/'profile',target_is_directory=True)
                save(target/'collection.json',dict(status='completed',side=label,spec_sha256=digest,source_sha256=spec['source_sha256'],groups='Roofline',
                    expected_kernel=discovery['selected'][0],original=str(directory),source_kind='new_capture'))
            module=HERE.parent/'skills/hgkda/scripts/compare_opprof.py';s=importlib.util.spec_from_file_location('demo_compare',module);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
            save(out/'comparison.json',m.compare(out))
    print(json.dumps(dict(status='completed',side=side,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())),flush=True)

if __name__=='__main__':main()
