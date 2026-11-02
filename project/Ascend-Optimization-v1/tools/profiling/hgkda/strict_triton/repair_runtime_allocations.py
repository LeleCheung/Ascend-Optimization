"""纠正new_empty分配的运行时误拒绝；保留旧检查，未提交的正式评测首次执行。"""
from pathlib import Path
from common import BATCH,HERE,PY,read,save,sha,load


def execute():
    module=load('strict_allocation_runner',HERE/'run.py')
    class AllocationRunner(module.Runner):
        def job(self,key,fn):return super().job(key+'@allocation-review-v2',fn)
        def qualify(self,op,source,leaf):
            output=source.parent/'runtime-audit-allocation-v2.json'
            if not output.exists():
                try:self.cmd('strict-runtime-allocation-v2-'+op,[PY,HERE/'runtime_audit.py','--operator',op,'--source',source,'--leaf',leaf,'--output',output],1200)
                except RuntimeError:
                    if not output.exists():raise
            value=read(output);assert value['source_sha256']==sha(source)
            return dict(status=value['status'],passed=value['passed'],total=value['total'],path=str(output))
    runner=AllocationRunner()
    for op in ['xlogy_scalar_tensor','native_group_norm']:
        row=runner.state['operators'].get(op,{})
        if row.get('runtime',{}).get('status')!='failed':continue
        old=read(row['runtime']['path'])
        failures=[c for c in old['cases'] if not c['passed']]
        assert failures and all('非许可ATen操作：aten.new_empty.default' in c.get('error','') for c in failures)
        saved=BATCH/'revisions'/op/'runtime-allocation-v1-state.json'
        if not saved.exists():save(saved,row)
        runner.state['operators'][op]={};runner.persist()
        runner.one(op)
        load('strict_allocation_report',HERE/'report.py').main()
    runner.state.update(status='completed',current_job=None);runner.persist()
