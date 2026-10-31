"""HG-KDA自有覆盖层的诊断预算；不改变KGS或正式评测合同。"""
import fcntl
import json
from pathlib import Path
import time


def admit(workspace, timeout_seconds):
    root=Path(workspace)/'.kernelgen'
    root.mkdir(parents=True,exist_ok=True)
    formal_results=len(list((root/'evals').glob('round-*/result.json')))
    allowed=min(10,2+2*formal_results)
    path=root/'hgkda-debug-budget.json'
    with (root/'hgkda-debug-budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(path.read_text()) if path.exists() else dict(requests=[])
        count=len(state['requests'])
        if count>=allowed:
            return dict(status='ERROR',reason='HGKDA_DEBUG_BUDGET',formal_results=formal_results,used=count,allowed=allowed,
                        instruction='诊断预算已用完。请根据已有日志修复候选，执行preflight_kernel和eval_round；正式评测后重新开放诊断预算。不要继续启动微基准。')
        bounded=min(120,int(timeout_seconds))
        state['requests'].append(dict(timestamp=time.time(),requested_seconds=timeout_seconds,admitted_seconds=bounded,formal_results=formal_results))
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8');temporary.replace(path)
        return dict(status='ADMITTED',timeout_seconds=bounded,used=count+1,allowed=allowed)
