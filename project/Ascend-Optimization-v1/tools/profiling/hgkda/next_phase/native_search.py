"""独立KG搜索控制：预算到期时让本worker已提交的正式评测短暂收尾。"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request
from catalog_snapshot import freeze_catalog

ROOT=Path("/data/hanle/ascend-optimization")

def read(path):return json.loads(path.read_text(encoding="utf8"))
def quote(parts):return " ".join(shlex.quote(str(x)) for x in parts)


def evaluation_grace(operations, server, elapsed, budget, grace=180):
    """只为当前工作区登记的同服务正式评测提供有限收尾窗口。"""
    if not budget < elapsed < budget+grace:return []
    return [item for item in operations.values()
            if item.get('kind')=='evaluate' and item.get('server_url')==server
            and item.get('operation_id')]


def native(self,op,attempt,seed,overlay,budget):
    suffix=self.batch.name+'-'+attempt
    workspace=self.q.runtime/'kg-runs'/(op+'-'+suffix)
    if workspace.exists():
        metadata=workspace/'.kernelgen/run-process.json'
        if metadata.exists() and read(metadata).get('finished_at'):return workspace
        raise self.module.Attention('已有worker未确认结束；不重复启动: '+str(workspace))
    catalog=freeze_catalog(self.batch/'catalog',self.batch/'catalog-snapshots'/(op+'-'+attempt),self.q.records.values())
    exports=dict(KG_SOURCE_ROOT=str(overlay),KG_RUN_ROOT=str(self.q.runtime/'kg-runs'),
        KG_CATALOG_PATH=str(catalog),KG_WALLTIME_PORT='19659',KG_PROFILE_PORT='19659',KG_MODEL='deepseek-flash',
        PYTHONDONTWRITEBYTECODE='1')
    body='\n'.join('export '+k+'='+shlex.quote(v) for k,v in exports.items())+'\n'+quote([
        'bash',self.q.project/'tools/evaluation/run-kg-operator-910b.sh',op,'profile',suffix,
        self.q.plan['passes'][int(attempt[-1])-1]['rounds'],seed])
    command='set -euo pipefail\nsource /usr/local/Ascend/cann-9.0.0/set_env.sh\nexport ASCEND_RT_VISIBLE_DEVICES=7\n'
    command+='export PYTHONPATH='+shlex.quote(':'.join(str(p) for p in [self.q.repo/'third_party/FlagGems/src',self.q.repo/'sources/kernelgen_server/client',self.q.repo/'sources/kernelgen_server',self.q.repo/'sources']))+'\n'
    command+='export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost\nunset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\n'+body
    log=self.q.runtime/(op+'-'+attempt+'.log')
    started=time.monotonic();seen={};cancelled=set();requested=None
    with log.open('ab') as stream:
        process=subprocess.Popen(['docker','exec','-i','tle_yy','bash','-s'],stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT)
        process.stdin.write(command.encode());process.stdin.close()
        while process.poll() is None:
            active=workspace/'.kernelgen/active-server-operations.json'
            operations={}
            if active.exists():
                operations=read(active).get('operations',{})
                for item in operations.values():
                    oid=item['operation_id'];seen.setdefault(oid,time.monotonic())
                    if (item['kind']=='profile' and item['server_url']==self.q.server and oid not in cancelled
                            and time.monotonic()-seen[oid]>self.q.plan['profile_budget_seconds']):
                        url=self.q.server+'/operations/'+oid
                        before=json.load(self.q.opener.open(url,timeout=15))
                        if before.get('kind')!='profile':raise self.module.Attention('操作身份改变')
                        after=json.load(self.q.opener.open(urllib.request.Request(url,method='DELETE'),timeout=20))
                        cancelled.add(oid)
                        self.module.save(self.batch/('watchdog-'+op+'-'+attempt+'-'+hashlib.sha256(oid.encode()).hexdigest()[:12]+'.json'),
                            dict(operation=item,before=before,after=after,reason='本批单次profile预算240秒；只取消本worker登记操作，不重启服务',time=self.module.now()))
            if time.monotonic()-started>budget and requested is None:
                deferred=[]
                for item in evaluation_grace(operations,self.q.server,time.monotonic()-started,budget):
                    status=json.load(self.q.opener.open(self.q.server+'/operations/'+item['operation_id'],timeout=15))
                    if status.get('kind')!='evaluate' or status.get('operation_id')!=item['operation_id']:
                        raise self.module.Attention('收尾评测操作身份不同')
                    if status.get('state') in {'QUEUED','RUNNING'}:
                        deferred.append(dict(operation=item,state=status))
                if deferred:
                    path=self.batch/('evaluation-grace-'+op+'-'+attempt+'.json')
                    if not path.exists():
                        self.module.save(path,dict(workspace=str(workspace),operations=deferred,
                            max_seconds=180,reason='搜索预算到期时保留已提交的完整评测收尾机会；不重发请求',time=self.module.now()))
                else:
                    result=subprocess.run(['docker','exec','-e','PYTHONPATH='+str(overlay), 'tle_yy',
                        str(ROOT/'runtime/kg-controller/venv/bin/kg'),'cancel',str(workspace),'--reason','夜间单项搜索预算到期，保留已通过轮次并转下项'],
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
                    self.module.save(self.batch/('budget-'+op+'-'+attempt+'.json'),dict(returncode=result.returncode,output=result.stdout.decode(errors='replace'),workspace=str(workspace)))
                    requested=time.monotonic()
            if requested and time.monotonic()-requested>180:
                raise self.module.Attention('已取消本worker但尚未确认退出；保留进程，不启动其他设备请求')
            self.module.save(self.batch/'heartbeat.json',dict(status='running',operator=op,attempt=attempt,
                workspace=str(workspace),elapsed_seconds=round(time.monotonic()-started),time=self.module.now()))
            time.sleep(10)
    metadata=workspace/'.kernelgen/run-process.json'
    if not metadata.exists() or not read(metadata).get('finished_at'):
        raise self.module.Attention('KG退出但完成元数据未知')
    self.q.idle();return workspace

