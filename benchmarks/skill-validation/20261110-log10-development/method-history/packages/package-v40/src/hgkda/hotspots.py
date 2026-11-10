"""筛选候选内核的逐硬件核指令热点。"""
import ast

def instruction_hotspots(listing, source, *, count=12):
    """只汇总源码中 jit 函数对应的指令热点，避免将输入构建 kernel 当作候选。"""
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
    return {'scope': '模拟器单指令观测，不能相加为真机总耗时', 'kernels': selected,
            'status': 'matched' if selected else 'no_candidate_kernel_match'}
