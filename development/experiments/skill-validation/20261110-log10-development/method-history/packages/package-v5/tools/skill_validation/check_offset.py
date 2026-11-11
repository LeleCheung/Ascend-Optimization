"""检查原地SELU在带偏移视图上的结果与周边存储；仅作为额外正确性证据。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('_checked_offset_source', args.source)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    rows = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        for layout in ['contiguous_offset', 'strided_offset', 'transposed_offset']:
            base = torch.linspace(-2, 2, 8192, device='npu:0', dtype=dtype).view(64, 128)
            before = base.clone()
            selected = torch.zeros_like(base, dtype=torch.bool)
            if layout == 'contiguous_offset':
                view = base[2:18, :]
                selected[2:18, :] = True
            else:
                view = base[2:18, 3:35:2]
                selected[2:18, 3:35:2] = True
                if layout == 'transposed_offset':
                    view = view.t()
            expected = torch.nn.functional.selu(view.clone())
            case = dict(dtype=str(dtype), layout=layout, storage_offset=view.storage_offset(),
                        shape=list(view.shape), stride=list(view.stride()))
            try:
                output = candidate.run(view)
                torch.npu.synchronize()
                assert output.data_ptr() == view.data_ptr() and output.stride() == view.stride()
                torch.testing.assert_close(view, expected, rtol=.015, atol=.015, equal_nan=True)
                assert torch.equal(base[~selected], before[~selected]), '视图以外的存储被改写'
                case['passed'] = True
            except Exception as error:
                case.update(passed=False, error=type(error).__name__+': '+str(error)[:1000])
            rows.append(case)
    result = dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  passed=sum(r['passed'] for r in rows), total=len(rows), cases=rows,
                  scope='额外偏移与存储保护检查；PyTorch仅提供参考，不属于候选计算；不更改正式合同')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(passed=result['passed'], total=result['total'])))


if __name__ == '__main__':
    main()
