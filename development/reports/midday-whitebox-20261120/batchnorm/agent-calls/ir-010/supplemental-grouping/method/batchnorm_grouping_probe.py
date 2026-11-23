"""Check changed statistical grouping against FP64 moments and a retained IR.

This supplemental diagnostic preserves the published contract and runs both
variants on identical, deterministic FP32 inputs. It records inherited accuracy
limits separately from newly introduced differences.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = read(args.spec)
    import torch
    import torch_npu
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    manifest_path = Path(spec['manifest'])
    runtime = imported('_grouping_override', manifest_path.parent / 'ir_override_runtime.py')
    source = runtime.install(manifest_path)
    assert sha(source) == spec['source_sha256']
    candidate = imported('_grouping_candidate', source)
    generator = torch.Generator().manual_seed(spec['seed'])
    records = []
    for spatial in spec['spatial_sizes']:
        shape = (8, 8, 1, spatial)
        noise = torch.randn(shape, generator=generator, dtype=torch.float32)
        for recipe in spec['recipes']:
            if recipe == 'normal':
                cpu = noise.clone()
            elif recipe == 'constant':
                cpu = torch.full(shape, 3.25, dtype=torch.float32)
            elif recipe == 'offset128':
                cpu = 128.0 + noise * 0.125
            elif recipe == 'offset4096':
                cpu = 4096.0 + noise * 0.125
            elif recipe == 'alternating':
                cpu = torch.arange(spatial).remainder(2).float().mul(2).sub(1)
                cpu = cpu.expand(shape).contiguous()
            else:
                raise ValueError(recipe)
            weight = torch.linspace(-1.5, 1.5, 8)
            bias = torch.linspace(-0.25, 0.25, 8)
            running_mean = torch.linspace(-2.0, 2.0, 8)
            running_var = torch.linspace(0.5, 2.0, 8)
            inputs = dict(input=cpu.to('npu:0'), weight=weight.to('npu:0'),
                          bias=bias.to('npu:0'), running_mean=running_mean.to('npu:0'),
                          running_var=running_var.to('npu:0'),
                          momentum=spec['momentum'], eps=spec['eps'])
            snapshots = {key: value.cpu().clone() for key, value in inputs.items()
                         if isinstance(value, torch.Tensor)}
            # CPU FP64 population moments provide an independent reference.
            x = cpu.double()
            mean = x.mean(dim=(0, 2, 3))
            variance = ((x - mean[None, :, None, None]) ** 2).mean(dim=(0, 2, 3))
            invstd = (variance + spec['eps']).rsqrt()
            output = weight.double()[None, :, None, None] * (
                x - mean[None, :, None, None]) * invstd[None, :, None, None]
            output += bias.double()[None, :, None, None]
            count = x.shape[0] * x.shape[2] * x.shape[3]
            momentum = spec['momentum']
            expected = [output.float(), mean.float(), invstd.float(),
                        torch.empty(0, dtype=torch.uint8),
                        ((1 - momentum) * running_mean.double() + momentum * mean).float(),
                        ((1 - momentum) * running_var.double()
                         + momentum * variance * count / (count - 1)).float()]
            actual = candidate.run(**inputs)
            torch.npu.synchronize()
            assert len(actual) == len(expected)
            row = dict(spatial=spatial, shape=list(shape), recipe=recipe,
                       input_sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                       outputs=[], passed=True)
            for index, (value, reference) in enumerate(zip(actual, expected)):
                observed = value.cpu()
                assert observed.shape == reference.shape and observed.dtype == reference.dtype
                difference = ((observed.double() - reference.double()).abs().max().item()
                              if reference.numel() else 0.0)
                metric = dict(index=index, max_abs_error=difference, passed=True)
                try:
                    torch.testing.assert_close(observed, reference, atol=spec['atol'],
                                               rtol=spec['rtol'], equal_nan=True)
                except AssertionError as error:
                    metric.update(passed=False, error=str(error)[:1800])
                    row['passed'] = False
                row['outputs'].append(metric)
            for key, value in snapshots.items():
                torch.testing.assert_close(inputs[key].cpu(), value, atol=0, rtol=0)
            records.append(row)
    manifest = read(manifest_path)
    verification = runtime.verify_receipts(manifest['receipt_directory'], manifest)
    target_binaries = {}
    for path in Path(manifest['receipt_directory']).glob('*.jsonl'):
        for line in path.read_text().splitlines():
            receipt = json.loads(line)
            if receipt.get('override_applied'):
                binary = next(item for item in receipt['artifacts'] if item['kind'] == 'npubin')
                assert sha(binary['path']) == binary['sha256']
                target_binaries[receipt['ast_sha256']] = binary['sha256']
    assert target_binaries == spec['expected_binaries'], 'Supplement changed the formal target binary'
    result = dict(status='completed', variant=spec['variant'], source_sha256=sha(source),
                  manifest_sha256=sha(manifest_path), spec_sha256=sha(args.spec),
                  probe_sha256=sha(__file__), passed=sum(row['passed'] for row in records),
                  total=len(records), rows=records, compiler_verification=verification,
                  verified_binaries=target_binaries,
                  scope='Independent FP64 diagnostic on fixed FP32 input recipes;published full '
                        'contract remains separate.Compare both variants to identify inherited limits.')
    Path(spec['output']).write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(variant=spec['variant'], passed=result['passed'], total=result['total'])))


if __name__ == '__main__':
    main()
