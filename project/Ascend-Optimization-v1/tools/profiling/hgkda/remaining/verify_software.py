"""只读核对固定FlagGems逐文件身份及KGS实际软件信息，不提交设备任务。"""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

PROJECT=Path(__file__).resolve().parents[4]
REPO=PROJECT.parents[1]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'


def main():
    locked=json.loads((BATCH/'software-lock.json').read_text())
    manifest=json.loads((REPO/'manifests/flaggems-master-files.json').read_text())
    assert manifest['upstream_commit']==locked['flaggems']['commit']
    for entry in manifest['files']:
        path=REPO/'third_party/FlagGems'/entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],str(path)
    services={};opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for port in [19655,19657,19659]:
        value=json.load(opener.open('http://127.0.0.1:%d/status'%port,timeout=15))
        assert value['software']==locked['service_software']
        services[str(port)]=dict(software=value['software'],target=value['target'],server_version=value['server_version'],timing=value['timing'])
    result=dict(status='verified',time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        flaggems_commit=manifest['upstream_commit'],flaggems_files=len(manifest['files']),services=services,
        service_software_checked=True,scope='静态文件和当前服务软件身份；物理卡7由控制器与既有服务映射约束')
    (BATCH/'baseline-verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],flaggems_files=result['flaggems_files'],services=3)))


if __name__=='__main__':main()
