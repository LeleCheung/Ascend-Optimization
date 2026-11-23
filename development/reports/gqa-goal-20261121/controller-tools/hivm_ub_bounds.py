"""Audit constant contiguous UB pointer views; dynamic views require host review."""
import argparse
import json
from pathlib import Path
import re


def audit(text, capacity_bytes):
    constants = {}
    ambiguous = set()
    for name, value in re.findall(r'(%[\w.$-]+)\s*=\s*arith.constant\s+(-?\d+)\s*:\s*i64\b', text):
        value = int(value)
        if name in constants and constants[name] != value:
            ambiguous.add(name)
        constants[name] = value
    views, unresolved = [], []
    widths = {'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
              'f16': 2, 'bf16': 2, 'f32': 4, 'f64': 8}
    for number, line in enumerate(text.splitlines(), 1):
        if 'hivm.hir.pointer_cast' not in line or '#hivm.address_space<ub>' not in line:
            continue
        match = re.search(r'pointer_cast\((%[\w.$-]+)\)\s*:\s*memref<([^,]+),\s*#hivm.address_space<ub>>', line)
        if not match:
            unresolved.append(dict(line=number, reason='Unsupported pointer or non-contiguous type'))
            continue
        name, shape = match.groups()
        parts = shape.split('x')
        element = parts[-1]
        if (name not in constants or name in ambiguous or element not in widths
                or not all(part.isdigit() for part in parts[:-1])):
            unresolved.append(dict(line=number, reason='Dynamic/ambiguous address, shape or element type'))
            continue
        size = widths[element]
        for dimension in parts[:-1]:
            size *= int(dimension)
        start = constants[name]
        views.append(dict(line=number, address_symbol=name, type=shape,
                          start_bytes=start, size_bytes=size, end_bytes=start + size))
    overflow = [view for view in views if view['size_bytes'] and
                (view['start_bytes'] < 0 or view['end_bytes'] > capacity_bytes)]
    return dict(status='static_ub_out_of_bounds' if overflow else
                'requires_dynamic_view_review' if unresolved or not views else 'static_pointer_bounds_verified',
                capacity_bytes=capacity_bytes, views=views, overflow=overflow, unresolved=unresolved,
                highest_static_end_bytes=max((v['end_bytes'] for v in views if v['size_bytes']), default=0),
                scope='Only constant-address contiguous UB pointer_cast views. Reused SSA constant names '
                      'with conflicting values are unresolved. This does not prove dynamic bounds, aliases, '
                      'library scratch use, liveness, event ordering or overall kernel correctness.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ir', type=Path, required=True)
    parser.add_argument('--capacity-bytes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.capacity_bytes > 0
    report = audit(args.ir.read_text(encoding='utf8'), args.capacity_bytes)
    import hashlib
    report.update(ir=str(args.ir.resolve()), ir_sha256=hashlib.sha256(args.ir.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print({key: report[key] for key in ['status', 'highest_static_end_bytes', 'capacity_bytes']})
    if report['overflow']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
