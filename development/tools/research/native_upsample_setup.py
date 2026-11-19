"""Freeze upstream bytes and probe native dependencies without changing a kernel."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path('/data/hanle/ascend-optimization')
RUN=ROOT/'runs/native-upsample-contrast-20261118'
REPO=ROOT/'Ascend-Optimization'


def main():
    RUN.mkdir(exist_ok=False)
    source=REPO/'development/dependencies/third_party/FlagGems/src/flag_gems/ops/upsample_bilinear2d.py'
    raw=source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='62a02c203a3e54a09f6479ca8526168bf1529821284cd1b42bc6cbdf3c980a12'
    (RUN/'upstream.py').write_bytes(raw)
    manifest={'upstream_commit':'d6a8eec473517a3d68157b208eb9c057eb1d4c50','upstream_source_sha256':hashlib.sha256(raw).hexdigest(),
              'budget_per_arm':{'generation_calls':8,'max_turns_per_call':48,'formal_candidates_per_call':1},
              'target':'upsample_bilinear2d-timing-1','formal_cases':24,'supplementary_each':30,
              'status':'native_import_probe','prior_best_source_provided':False}
    (RUN/'experiment.json').write_text(json.dumps(manifest,indent=2))
    probe="""import importlib.util, json, sys
from pathlib import Path
import torch, torch_npu
import flag_gems
p=Path(__file__).parent/'upstream.py'
s=importlib.util.spec_from_file_location('native_upsample_probe',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print(json.dumps({'flag_gems':flag_gems.__file__,'torch':torch.__version__,'entrypoint':m.upsample_bilinear2d.__name__}),flush=True)
"""
    (RUN/'probe.py').write_text(probe)
    deps=REPO/'development/dependencies/third_party/FlagGems/src'
    shell='source /usr/local/Ascend/cann-9.0.0/set_env.sh\nexport ASCEND_RT_VISIBLE_DEVICES=7\nexport PYTHONPATH='+shlex.quote(str(deps))+'\nexec /usr/local/python3.11.15/bin/python3.11 '+shlex.quote(str(RUN/'probe.py'))
    command=['docker','exec','-w',str(RUN),'tle_yy','bash','-c',shell]
    with (RUN/'probe.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    manifest['probe_returncode']=result.returncode
    (RUN/'experiment.json').write_text(json.dumps(manifest,indent=2))
    print((RUN/'probe.log').read_text())


if __name__=='__main__':main()
