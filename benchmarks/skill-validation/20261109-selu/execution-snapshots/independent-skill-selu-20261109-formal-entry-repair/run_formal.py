import hashlib,json,subprocess,sys,time
from pathlib import Path
b=Path('/data/hanle/ascend-optimization');repo=b/'Ascend-Optimization';r=b/'runs/independent-skill-selu-20261109-v1'
digest='3cec1212c3b7630bbf548188693f4cd9fef8747ee78cb1f822a42dcf685a23ed'
attempt=r/'attempts'/digest
for i in range(90):
 if (attempt/'preflight.json').exists() and (r/'agent-calls/07-kgs-entry-compatibility/model-usage.json').exists():break
 time.sleep(5)
else:raise RuntimeError('等待入口修复预检超时，未发送正式请求')
p=json.loads((attempt/'preflight.json').read_bytes());assert p['status']=='passed'
first=r/'formal-f3502fad5259';assert json.loads((first/'state.json').read_bytes())['status']=='completed'
assert (first/'service-stop.json').exists()
old=(r/'attempts/f3502fad52595cc909e47a5a158ce16edcdfdd8aa11b2f5c7657ff367e46d288/candidate.py').read_bytes()
source=attempt/'candidate.py';raw=source.read_bytes()
expected=old.replace(b'def selu_(*args, **kwargs):',b'def _selu_entry(*args, **kwargs):').replace(b'return selu_(A)',b'return _selu_entry(A)')
assert raw==expected,'入口改名以外还有变更，需要重新完整审查'
review=json.loads((r/'source-review.json').read_bytes());review['parent_source_sha256']=review['source_sha256'];review['source_sha256']=digest
review['functions_reviewed']=[('_selu_entry' if x=='selu_' else x) for x in review['functions_reviewed']]
review['review_notes'].append('仅将Python辅助入口私有化，设备kernel及所有调用路径逐字节不变；offset 9/9与父版本的设备代码一致性另行绑定。')
review_path=r/'source-review-entry-repaired.json';review_path.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
equivalence=dict(source_sha256=digest,parent_source_sha256=review['parent_source_sha256'],
 expected_rename_matches=True,device_kernels_unchanged=True,
 offset_evidence=dict(path=str(r/'offset-check-final.json'),sha256=hashlib.sha256((r/'offset-check-final.json').read_bytes()).hexdigest()),
 scope='仅Python辅助符号改名；额外偏移9/9复用完全相同设备路径，不重复提交同类正确性测量')
(r/'entry-equivalence.json').write_text(json.dumps(equivalence,ensure_ascii=False,indent=2)+'\n')
subprocess.run([sys.executable,str(repo/'tools/skill_validation/evaluate.py'),'--config',str(r/'config-entry-repair.json'),
 '--source',str(source),'--review',str(review_path),'--port','19693'],check=True)
