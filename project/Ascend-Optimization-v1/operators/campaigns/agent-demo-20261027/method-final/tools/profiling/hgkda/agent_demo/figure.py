"""复用已校验Roofline点和中文绘图，不再调用设备。"""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--verify-existing',action='store_true');a=p.parse_args();b=a.batch
    evidence=json.loads((b/'agent-input.json').read_bytes());feedback=json.loads((b/'feedback/facts.json').read_bytes())
    cases=[]
    for pair in evidence['pairs']:
        for side,row in pair['records'].items():
            spec=dict(row['spec']);spec['side']='parent' if side=='candidate' else side
            cases.append(dict(spec=spec,diagnostic=row['roofline']))
    for pair in feedback['pairs']:
        row=pair['facts']['candidate'];spec=dict(row['source_spec']);spec['side']='candidate';cases.append(dict(spec=spec,diagnostic=row['roofline']))
    output=b/'figure-data.json'
    if a.verify_existing:
        assert json.loads(output.read_bytes())==dict(cases=cases)
    else:
        assert not output.exists();output.write_text(json.dumps(dict(cases=cases),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        s=importlib.util.spec_from_file_location('demo_plot',HERE.parent/'roofline/plot_chinese.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        sys.argv=[str(HERE/'figure.py'),'--analysis',str(output),'--output',str(b/'图表')];m.main()
    files=[output,*sorted((b/'图表').glob('*'))];assert len(files)>=4
    marker=b/'figure-completed.json';assert not marker.exists()
    marker.write_text(json.dumps(dict(status='completed',source='同批新采集，复用数值解析器与绘图',files=[dict(path=p.relative_to(b).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),ensure_ascii=False,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':main()
