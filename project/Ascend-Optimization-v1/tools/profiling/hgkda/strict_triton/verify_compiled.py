"""在原队列结束后串行提取实际Triton编译产物；空trace不作设备核验成功。"""
import argparse
import json
from pathlib import Path
from common import BATCH,HERE,NEXT,ROOT,PY,read,save,sha,load
from gate import admit


def validate_manifest(manifest,source):
    value=read(manifest)
    assert value['spec']['source_sha256']==sha(source)
    assert value.get('entries'),'未捕获实际Triton编译核'
    artifacts={str(manifest):sha(manifest)}
    kernels=[]
    for entry in value['entries']:
        kinds=set()
        for item in entry['artifacts']:
            path=Path(item['path']);assert path.stat().st_size>0
            assert sha(path)==item['sha256']
            artifacts[str(path)]=item['sha256'];kinds.add(item['kind'])
        assert 'ttir' in kinds,(entry['symbol'],kinds)
        assert 'npubin' in kinds,(entry['symbol'],kinds)
        kernels.append(dict(symbol=entry['symbol'],artifact_kinds=sorted(kinds),metadata=entry['metadata']))
    return dict(status='verified',source_sha256=sha(source),manifest=str(manifest),kernels=kernels,
        artifacts_sha256=artifacts,
        scope='代表输入实际调用返回的Triton编译对象，绑定TTIR及npubin；与逐路径源码、依赖及运行时审查合用。不是所有输入的硬件任务trace。')


def execute(operators=None):
    state=read(BATCH/'state.json')
    assert state['status']=='completed','完整队列尚未结束，禁止并行设备采集'
    if operators is None or 'transform_bias_rescale_qkv' in operators:
        load('strict_restore_host',HERE/'restore_host_glue.py').execute()
    runner=load('strict_verify_runner',HERE/'run.py').Runner()
    state=read(BATCH/'state.json')
    index=read(BATCH/'candidate-index.json');verification=BATCH/'compiled-verification.json'
    progress=read(verification) if verification.exists() else dict(status='running',operators={})
    progress['status']='running';save(verification,progress)
    for op in operators or list(index):
        source=Path(index[op]['source']);admit(source,source.parent/'review.json')
        row=state['operators'].get(op,{})
        if row.get('runtime',{}).get('status')!='passed':continue
        leaf=BATCH/runner.q.records[op]['catalog_leaf']
        cases=[json.loads(x) for x in (leaf/'correctness.jsonl').read_text().splitlines() if x.strip()]
        case=next((v for v in cases if v.get('inputs')),cases[0])
        directory=source.parent/'actual-compiled-v1';manifest=directory/'ir/manifest.json'
        directory.mkdir(parents=True,exist_ok=True)
        spec=directory/'spec.json';payload=dict(operator=op,source=str(source),source_sha256=sha(source),leaf=str(leaf),case=case)
        if spec.exists():assert read(spec)==payload
        else:save(spec,payload)
        try:
            if not manifest.exists():
                assert not (directory/'ir').exists(),'不覆盖未完成编译目录，先人工检查'
                runner.wait()
                runner.cmd('strict-compiled-'+op,[PY,NEXT/'driver.py','--spec',spec,'--side','candidate','--ir',directory/'ir'],900)
            proof=validate_manifest(manifest,source)
            output=directory/'verification.json';save(output,proof)
            row['compiled_path']=dict(status='verified',path=str(output),source_sha256=sha(source),kernels=[k['symbol'] for k in proof['kernels']])
            progress['operators'][op]=row['compiled_path']
            print('COMPILED_VERIFIED',op,row['compiled_path']['kernels'],flush=True)
        except Exception as error:
            row['compiled_path']=dict(status='failed',error=type(error).__name__+': '+str(error),source_sha256=sha(source))
            progress['operators'][op]=row['compiled_path']
            print('COMPILED_FAILED',op,row['compiled_path']['error'],flush=True)
        state['operators'][op]=row;save(BATCH/'state.json',state);save(verification,progress)
        load('strict_verified_report',HERE/'report.py').main()
    runner.wait();progress['status']='completed';save(verification,progress)
    load('strict_verified_report',HERE/'report.py').main()


def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--operator',action='append');a=p.parse_args()
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        execute(a.operator)


if __name__=='__main__':main()
