"""队列全部退出后归档本项目日志，核验服务并生成中文简报；不提交设备任务。"""
import argparse
import fcntl
import re
import shutil
import subprocess
import time
from pathlib import Path
from common import BATCH,HERE,ROOT,load,read,save,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    runtime=ROOT/'runs'/BATCH.name
    if args.wait:
        while subprocess.run(['tmux','has-session','-t','hanle-cann-priority-20261102-followup'],
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE).returncode==0:
            time.sleep(10)
    assert read(runtime/'followup-state.json')['status']=='completed','后续队列仍在运行'
    state=read(BATCH/'next-state.json')
    assert state['status']=='phase_completed' and state.get('current_job') is None,'正式队列尚未收尾'
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=load('cann_finalize_queue',HERE.parent/'run-nightly.py').Queue(BATCH.name,ROOT,'tle_yy')
        q.idle()
        load('cann_finalize_report',HERE/'report.py').main()
        report=read(BATCH/'delivery.json')
        assert not report['pending_requests'] and not report['audit_issues'],'证据仍有未决项'
        logs={}
        origins=list(runtime.glob('*.log'))+list(runtime.glob('*.json'))
        origins=[p for p in origins if p.name!='finalize.log']
        extra=ROOT/'runtime/cann-priority-rad2deg-device-v1/build.log'
        if extra.exists():origins.append(extra)
        sensitive=re.compile(rb'(?<![A-Za-z0-9_-])(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9_-]{24,})')
        for source in sorted(origins):
            raw=source.read_bytes()
            if sensitive.search(raw):raise RuntimeError('日志含疑似凭据，保留本机待处理：'+str(source))
            name='rad2deg-device-build.log' if source==extra else source.name
            target=BATCH/'runtime-logs'/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==raw,'最终日志已冻结，请新建归档版本'
            else:shutil.copyfile(source,target)
            logs[str(target)]=dict(origin=str(source),sha256=sha(target),bytes=len(raw))
        save(BATCH/'runtime-logs/manifest.json',dict(status='verified',files=logs))
        new=[r for r in report['rows'] if r.get('selected_new')]
        assert all(r['selected_new'].get('paired_paths',{}).get('status')=='captured' for r in new), '有效新候选仍缺同输入设备路径'
        lines=['# 2026-11-02 CANN优先优化结果','',
            '以原30项输入、容差及KGS完整调用计时继续评测。原PyTorch参考常规达标24项另列。',
            '本批%d项CANN参考通过资格，%d项候选相对合格CANN参考超过0.8×。'%(report['cann_references_qualified'],report['cann_accepted']),
            '', '|新增有效候选|相对CANN|相对PyTorch|相对本批起点的实际加速|完整评测|',
            '|---|---:|---:|---:|---|']
        for row in new:
            item=row['selected_new'];p=item['pytorch'];c=item['cann']
            lines.append('|%s|%.3f×|%.3f×|%.3f×|%d/%d|'%(row['operator'],c['score'],p['score'],
                item['candidate_latency_gain'],*c['complete']))
        lines+=['','本批新跨过CANN验收线：'+('、'.join(report['newly_crossed_cann']) or '无')+'。',
            '本批三项选中源码均由Codex直接实现：Hardswish out复用自写Ascend C并改接out／大分块，Rad2deg自写常量乘设备核，QKV将既有Triton实现的宿主包装移至C++。历史KG Triton计算／布局路径保留，来源见proposal／manifest。',
            'CANN单接口、布局／dtype兼容、随机状态适配和组合实现的覆盖数见delivery.json；不能将完整调用加速直接解释为设备kernel加速。',
            '', '## 尚需继续处理','']
        for row in report['rows']:
            if not row.get('cann_accepted'):
                lines.append('- `'+row['operator']+'`：'+row['cann_status']+'；原状态：'+str(row['previous_status'])+'。')
        selection=BATCH/'search-selection.json'
        if selection.exists():
            selected=read(selection)['operators']
            lines+=['','本批额外KG／CC／DeepSeek搜索对象%d项；已通过本轮代码优化过线的对象不重复搜索。'%len(selected)]
        lines+=['','## 复现入口','',
            '在910B项目目录使用宿主控制端Python执行；已有正式请求只读取结果，不重复提交：','',
            '```bash',str(ROOT/'runtime/kg-controller/venv/bin/python')+' tools/profiling/hgkda/cann_priority/launch.py --phase all','```','',
            '后续参考补充与剩余优化由followup.py串行执行。构建源码、候选、参考、合同、完整请求、逐case结果、补测和设备trace均在本批目录。',
            '交付核验%d项，归档本项目日志%d份。三项KGS服务已确认healthy且idle，物理卡7项目锁已可用。'%(len(report['evidence_sha256']),len(logs))]
        (BATCH/'结果简报.md').write_text(chr(10).join(lines)+chr(10),encoding='utf8')
        save(BATCH/'final-checks.json',dict(status='verified',services='19655/19657/19659 healthy and idle',
            physical_card=7,exclusive_project_lock_available=True,followup='completed',
            evidence_count=len(report['evidence_sha256']),log_count=len(logs),
            delivery_sha256=sha(BATCH/'delivery.json'),brief_sha256=sha(BATCH/'结果简报.md')))
        print('FINAL_VERIFIED',report['cann_references_qualified'],report['cann_accepted'],report['newly_crossed_cann'])


if __name__=='__main__':main()
