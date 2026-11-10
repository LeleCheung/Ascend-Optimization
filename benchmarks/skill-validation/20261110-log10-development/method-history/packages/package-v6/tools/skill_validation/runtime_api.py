"""只读提取本机Triton后端启动接口及版本；不生成或执行目标算子实现。"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    import torch,torch_npu,triton
    import triton.backends.ascend.driver as driver
    import triton.backends.ascend.backend_register as backend
    records=[]
    for module in [driver,backend]:
        path=Path(inspect.getfile(module));raw=path.read_bytes();lines=raw.decode('utf8').splitlines()
        selected=set()
        for i,line in enumerate(lines):
            if any(key in line for key in ['def make_launcher','PyArg_ParseTuple(args','def get_current_stream',
                                           '_npu_getCurrentRawStream','OpCommand','SetCustomHandler','TRITON_ENABLE_TASKQUEUE']):
                selected.update(range(max(0,i-5),min(len(lines),i+24)))
        records.append(dict(module=module.__name__,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
             excerpts=[dict(line=i+1,text=lines[i]) for i in sorted(selected)]))
    result=dict(torch=torch.__version__,triton=triton.__version__,sources=records,
        scope='当前运行环境的只读接口事实；不包含目标算子候选实现，ABI仍需当前候选预检')
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='completed',versions=dict(torch=result['torch'],triton=result['triton']))))


if __name__=='__main__':main()
