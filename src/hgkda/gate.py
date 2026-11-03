"""严格 Triton 路径的静态筛查与源码审查身份检查。"""
import ast
import hashlib

FORBIDDEN_METHODS = {'contiguous', 'clone', 'copy_', 'fill_', 'zero_', 'flatten',
                     'reshape', 'to', 'cpu', 'cuda', 'numpy', 'resolve_conj', 'resolve_neg'}
TORCH_METADATA = {'empty', 'empty_like', 'empty_strided', 'device', 'dtype', 'is_tensor',
                  'Tensor', 'Size', 'finfo', 'iinfo', 'numel', 'get_default_dtype'}


def inspect_text(text):
    tree = ast.parse(text)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split('.')[0]] = item.name
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or '') + '.' + item.name

    def canonical(node):
        name = ast.unparse(node)
        root, separator, rest = name.partition('.')
        return aliases.get(root, root) + (separator + rest if separator else '')

    issues, kernels, calls = [], [], set()
    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            if any(canonical(d.func if isinstance(d, ast.Call) else d) == 'triton.jit'
                   for d in node.decorator_list):
                kernels.append(node.name)
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and canonical(sub.func).startswith('torch.'):
                        issues.append({'line': sub.lineno, 'call': canonical(sub.func), 'reason': '核内出现 PyTorch 调用'})
                return
            self.generic_visit(node)

        def visit_Call(self, node):
            name = canonical(node.func)
            calls.add(name)
            forbidden = name in {'eval', 'exec', '__import__'} or name.startswith(('ctypes.', 'subprocess.', 'acl.'))
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_METHODS:
                forbidden = True
            if name.startswith('torch.'):
                tail = name.removeprefix('torch.')
                if tail not in TORCH_METADATA and not tail.startswith(('npu.current_', 'cuda.current_')):
                    forbidden = True
            if forbidden:
                issues.append({'line': node.lineno, 'call': name, 'reason': '宿主计算、搬运或外部执行入口需要拒绝或单独审查'})
            self.generic_visit(node)
    Visitor().visit(tree)
    return {'source_sha256': hashlib.sha256(text.encode('utf8')).hexdigest(), 'issues': issues,
            'triton_kernels': kernels, 'host_calls': sorted(calls), 'static_passed': not issues,
            'admission': 'review_required', 'scope': '已知入口的 AST 筛查，不是完整路径证明'}


def admit(text, review):
    result = inspect_text(text)
    if review.get('source_sha256') != result['source_sha256']:
        raise ValueError('审查记录与提交源码不同')
    if review.get('status') != 'reviewed_strict_triton' or not all(
            review.get(key) is True for key in ['all_branches_reviewed', 'dependencies_reviewed']):
        raise ValueError('缺少严格 Triton 的完整调用路径审查')
    if result['issues']:
        raise ValueError('候选存在需要处理的静态风险：' + str(result['issues']))
    result['admission'] = 'review_bound'
    return result
