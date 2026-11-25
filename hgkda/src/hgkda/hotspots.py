"""Select instruction hotspots per hardware core for candidate kernels."""
import ast

def instruction_hotspots(listing, source, *, count=12):
    """Aggregate only instructions mapped to source JIT functions, excluding input-construction kernels."""
    symbols = {n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)
               and any('jit' in ast.unparse(d) for d in n.decorator_list)}
    selected = []
    for kernel in listing.get('kernels', []):
        name = kernel.get('name', '')
        if not any(name == s or name.startswith(s + '_') for s in symbols):
            continue
        for core in kernel.get('cores', []):
            rows = [r for r in core.get('instructions', []) if isinstance(r, dict)]
            top = sorted(rows, key=lambda r: (r.get('running_time_us') or 0,
                                              r.get('cycles') or 0), reverse=True)[:count]
            selected.append({'kernel': name, 'core': core.get('name'),
                             'listed_instruction_count': core.get('listed_instruction_count'),
                             'mapped_instruction_count': core.get('mapped_instruction_count'),
                             'hotspots': top})
    return {'scope': 'Simulator instruction observations cannot be summed into whole-call device latency', 'kernels': selected,
            'status': 'matched' if selected else 'no_candidate_kernel_match'}
