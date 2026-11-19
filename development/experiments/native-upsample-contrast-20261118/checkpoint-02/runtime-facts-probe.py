"""Read actual native launch metadata; no target kernel is executed."""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from flag_gems import runtime
from flag_gems.utils import triton_lang_extension

r=Path('/data/hanle/ascend-optimization/runs/native-upsample-contrast-20261118')
rows=[]
for config in runtime.get_tuned_config('upsample_bilinear2d'):
    rows.append({'kwargs':config.kwargs,'num_warps':config.num_warps,'num_stages':config.num_stages,'num_ctas':config.num_ctas})
p=Path(triton_lang_extension.__file__)
report={'source_sha256':hashlib.sha256((r/'upstream.py').read_bytes()).hexdigest(),'configs':rows,
        'program_id_semantics':'tl.program_id(axis).to(tl.int64)',
        'program_id_module':str(p),'program_id_module_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'scope':'Native metadata query only; no target kernel run or performance claim.'}
(r/'native-runtime-facts.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
