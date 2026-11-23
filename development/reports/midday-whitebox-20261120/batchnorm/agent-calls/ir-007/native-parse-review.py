from pathlib import Path
import sys,json,hashlib,traceback
sys.path.insert(0,'/data/hanle/ascend-optimization/runs/midday-whitebox-20261120/tools')
import torch,torch_npu
import triton.compiler.compiler as compiler
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from ir_lowering_review import target_fields
run=Path('/data/hanle/ascend-optimization/runs/midday-whitebox-20261120/batchnorm');call=run/'agent-calls/ir-007';inventory=json.loads((run/'workspace/ir-inventory.json').read_bytes());entry=inventory['entries'][0];f=call/'output-ir'/entry['path']
assert not torch.npu.is_initialized()
c=ir.context();ir.load_dialects(c);backend=compiler.make_backend(GPUTarget(**target_fields(entry['target'])));backend.load_dialects(c);compiler.buffer_ir.load_dialects(c);compiler.ascend_ir.load_dialects(c);compiler._flagprism.load_dialects(c)
result=dict(source_sha256=inventory['source_sha256'],candidate_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),npu_initialized=False)
try:
 module=ir.parse_mlir_module(str(f),c)
 result.update(status='native_parse_passed')
except Exception as e:result.update(status='native_parse_failed',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
with (call/'syntax-review.json').open('x') as out:json.dump(result,out,indent=2)
print(json.dumps(result))
