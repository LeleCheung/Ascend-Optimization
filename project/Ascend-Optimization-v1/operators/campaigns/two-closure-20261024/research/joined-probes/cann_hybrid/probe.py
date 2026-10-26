"""隔离进程检查原生扫描混合路线；原默认与官方确定性条件分别保存。"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib, importlib.util, json, time
import torch, torch_npu

def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

batch = Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024')
out = batch / 'research/cann-hybrid'
out.mkdir(parents=True, exist_ok=True)
source = batch / 'knowledge/cumulative-cann-sync-probe.py'
candidate = load('cann_sync_candidate', source)
oracle = load('cann_sync_oracle', batch / 'catalog/ops/cumulative_trapezoid/oracle.py')
cases = [json.loads(line) for line in (batch / 'catalog/ops/cumulative_trapezoid/timing.jsonl').read_text().splitlines()]
rows = []
for mode in ['default', 'torch_deterministic']:
    torch.use_deterministic_algorithms(mode == 'torch_deterministic')
    for case in cases:
        args = oracle.gen_inputs(case, 'npu:0')
        expected = oracle.run(**args)
        torch.npu.synchronize()
        row = dict(mode=mode, case=case['name'], tolerance=case['tolerance'])
        try:
            got = candidate.run(**args)
            torch.npu.synchronize()
            a, b = got.cpu().float(), expected.cpu().float()
            tol = case['tolerance']
            wrong = (a-b).abs() > tol['atol'] + tol['rtol'] * b.abs()
            row.update(passed=not bool(wrong.any()), mismatches=int(wrong.sum()),
                max_absolute=float((a-b).abs().max()))
            torch.testing.assert_close(got.cpu(), expected.cpu(), atol=tol['atol'], rtol=tol['rtol'])
        except Exception as error:
            row.update(passed=False, error=type(error).__name__ + ': ' + str(error)[-2000:])
        rows.append(row)
        print('CANN_HYBRID', json.dumps(row), flush=True)
        (out / 'results.json').write_text(json.dumps(dict(
            scope='同步混合数值探针，非正式性能；确定性条件单列，默认合同保持不变',
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), rows=rows), ensure_ascii=False, indent=2)+'\n')
