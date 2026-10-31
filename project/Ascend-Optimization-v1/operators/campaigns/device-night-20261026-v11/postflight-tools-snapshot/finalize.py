"""评测终态后的只读核验、层级证据与中文交付；不再调用模型或使用NPU。"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def finalize(batch):
    state=json.loads((batch/'state.json').read_text())
    if state['status']!='completed': raise ValueError('主评测尚未完整完成，不生成交付')
    delivery=batch/'delivery'
    if delivery.exists(): raise ValueError('交付目录已存在，不覆盖核验或报告')
    delivery.mkdir()
    record=dict(status='running',started_at=now(),scope='采集后核验，不参与冻结模型诊断或选版')
    save(batch/'postflight.json',record)
    try:
        jobs=[['audit.py','--batch-dir',str(batch),'--output',str(delivery/'audit.json')],
              ['audit_artifacts.py','--batch-dir',str(batch),'--output',str(delivery/'artifact-audit.json')],
              ['evidence_layers.py','--batch-dir',str(batch),'--output',str(delivery/'layered-evidence.json')],
              ['report.py','--batch-dir',str(batch),'--audit',str(delivery/'audit.json'),
               '--artifacts-audit',str(delivery/'artifact-audit.json'),'--output',str(delivery)]]
        for arguments in jobs:
            subprocess.run([sys.executable,str(HERE/arguments[0])]+arguments[1:],check=True)
        if (batch/'environment.json').exists(): shutil.copyfile(batch/'environment.json',delivery/'environment.json')
        shutil.copyfile(batch/'timing-scope-evidence/scope.json',delivery/'timing-scope-correction.json')
        ledger=dict(schema='hgkda-device-delivery-v1',batch=batch.name,
            state_sha256=sha(batch/'state.json'),plan_sha256=sha(batch/'plan.json'),
            selection_sha256=sha(batch/'selection-frozen.json'),frozen_method_sha256=sha(batch/'frozen-method.json'),
            postflight_tools={name:sha(HERE/name) for name in ['finalize.py','audit.py','audit_artifacts.py','evidence_layers.py','report.py']},
            files={p.name:sha(p) for p in delivery.iterdir() if p.is_file()},
            policy='冻结的system-snapshot保留原字节；此工具清单标识事后核验版本，不替换冻结方法')
        save(delivery/'ledger.json',ledger)
        record.update(status='completed',finished_at=now(),ledger_sha256=sha(delivery/'ledger.json'))
    except BaseException as error:
        record.update(status='needs_attention',finished_at=now(),error=type(error).__name__+': '+str(error));raise
    finally: save(batch/'postflight.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch-dir',type=Path,required=True)
    finalize(parser.parse_args().batch_dir.resolve())
