#!/usr/bin/env python3
"""从已核验交付建立上采样与累计积分新批次，冻结原合同及新方法预算。"""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path

TOOLS=Path(__file__).resolve().parent
PROJECT=TOOLS.parents[2]
REPO=PROJECT.parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def prepare(name):
 old=PROJECT/'operators/campaigns/remaining-five-20261023-v2'
 batch=PROJECT/'operators/campaigns'/name
 assert not batch.exists();batch.mkdir()
 plan=json.loads((old/'plan.json').read_text(encoding='utf-8'))
 final=json.loads((old/'final-delivery.json').read_text(encoding='utf-8'))
 rows=[]
 for op in ['upsample_bilinear2d','cumulative_trapezoid']:
  record=dict(next(r for r in plan['records'] if r['operator']==op))
  leaf=batch/record['catalog_leaf'];leaf.mkdir(parents=True)
  for filename,digest in record['contract_sha256'].items():
   source=old/record['catalog_leaf']/filename;assert sha(source)==digest;shutil.copyfile(source,leaf/filename)
  source=old/'deliverables/reviewed/upsample_bilinear2d/main.py' if op=='upsample_bilinear2d' else old/record['seed']
  if op=='upsample_bilinear2d':assert sha(source)=='2054ca728198ae2a395addf604834a00f67f4e5352acc84adb93250457bee84b'
  target=batch/'seeds'/(op+'.py');target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
  record.update(seed=target.relative_to(batch).as_posix(),source_sha256=sha(target),seed_origin=str(source))
  rows.append(record)
 catalog=json.loads((old/'catalog/manifest.json').read_text(encoding='utf-8'));catalog['name']=name;save(batch/'catalog/manifest.json',catalog)
 knowledge=batch/'knowledge';knowledge.mkdir()
 for name in ['累计积分参考自洽反例已实测.md','累计积分九项舍入与准确参考源码.md','上采样第二轮建议.md','上采样连续输入块与启动建议.md','完整调用的启动层开销.md']:
  shutil.copyfile(old/'knowledge'/name,knowledge/name)
 save(knowledge/'previous-delivery.json',final)
 brief='''# 两项收尾任务

只聚焦upsample_bilinear2d、cumulative_trapezoid。上采样从完整有效2054ca源码约0.340×继续，目标完整调用>0.8×。累计积分参考FP16非末轴存在重复调用超过原容差反例，先定位最小形状和官方确定性选项；诊断选项仅用于隔离进程，不改变原正式合同、容差或参考。

本轮优先验证新的数据流：CANN extension.extract_slice/insert_slice替代水平gather；标量行基址连续读取；独立上下两行/相邻输出行复用；可分离两kernel；必要时本机TLE DSA显式UB子视图。每条路线先小探针数值和完整调用收益，再完整门禁及一次正式评测。此前连续输入块、固定stride、位打包均未带来有效收益，不能重复当成新成果。

不要以裸kernel少、program少、理论搬运少直接认定快。小输入候选约109微秒，参考约33微秒，大输入约630微秒，参考约206微秒，必须同时处理launch参数和设备路径。libentry可正常验证，保持标准kernel[grid]、当前stream和独立输出，禁调被优化PyTorch/FlagGems算子或缓存输入/结果。

正确性包含原correctness/timing全部输入、stride与显式scale、两次及换stream；上采样另36项特殊门禁。一次正式完整复验，不反复测同字节挑分。有限方法预算仍未达标时保留真实最佳、具体编译/执行限制与累计积分最小参考反例。Codex实质诊断/修改与CC/DeepSeek生成分别记录。

固定FlagGems d6a8eec/FlagTree c286cba6、CANN9.0、tle_yy、物理卡7、KGS19659、串行卡锁；不重启服务、不动他人进程或共享后端。导入oracle关闭字节码写入，保持严格catalog洁净。
'''
 (knowledge/'夜间任务.md').write_text(brief,encoding='utf-8')
 toolnames=['run-two-closure.py','run-remaining-five.py','run-endurance.py','run-nightly.py','check-remaining-five.py','check-remaining-special.py','check-endurance.py','observe-endurance.py','build-kg-overlay.py']
 save(knowledge/'tool-sha256.json',{n:sha(TOOLS/n) for n in toolnames})
 save(knowledge/'supplemental-tool-sha256.json',{'check-remaining-special.py':sha(TOOLS/'check-remaining-special.py')})
 plan.update(schema_version='hgkda-two-closure-v1',records=rows,duration_hours=6,
             passes=[dict(rounds=5,budget_seconds=3600),dict(rounds=4,budget_seconds=3000)],
             base_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip(),parent_campaign=old.name)
 save(batch/'plan.json',plan)
 (batch/'README.md').write_text('# 上采样与累计积分收尾批次\n\n从上采样0.340×有效源码继续优化；累计积分先复现并缩小参考不稳定问题。任务见knowledge/夜间任务.md。原合同不变，一次正式完整评测，补充所有性能输入数值、布局、scale和stream。运行不等于达标，最终看交付清单。仅物理卡7和KGS19659串行执行，tmux持久运行。\n',encoding='utf-8')
 return batch

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True)
 print(prepare(p.parse_args().batch))
