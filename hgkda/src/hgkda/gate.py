"""Static screening and source-review identity checks for strict Triton paths."""
import ast
import hashlib

FORBIDDEN_METHODS = {'contiguous', 'clone', 'copy_', 'fill_', 'zero_', 'flatten',
                     'reshape', 'to', 'cpu', 'cuda', 'numpy', 'resolve_conj', 'resolve_neg'}
TORCH_METADATA = {'empty', 'empty_like', 'empty_strided', 'device', 'dtype', 'is_tensor',
                  'Tensor', 'Size', 'finfo', 'iinfo', 'numel', 'get_default_dtype',
                  'broadcast_shapes', 'view_as_real', 'view_as_complex', '_C._is_alias_of',
                  'npu.device_count', 'npu.get_device_properties',
                  'autograd._profiler_enabled'}


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
                        issues.append({'line': sub.lineno, 'call': canonical(sub.func), 'reason': 'PyTorch call inside a kernel'})
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
                issues.append({'line': node.lineno, 'call': name, 'reason': 'Host computation, data movement, or external execution requires rejection or separate review'})
            self.generic_visit(node)
    Visitor().visit(tree)
    return {'source_sha256': hashlib.sha256(text.encode('utf8')).hexdigest(), 'issues': issues,
            'triton_kernels': kernels, 'host_calls': sorted(calls), 'static_passed': not issues,
            'admission': 'review_required', 'scope': 'AST screening of known entry points, not a complete execution-path proof'}


def admit(text, review):
    result = inspect_text(text)
    if review.get('source_sha256') != result['source_sha256']:
        raise ValueError('Review record belongs to different submitted source')
    if review.get('status') != 'reviewed_strict_triton' or not all(
            review.get(key) is True for key in ['all_branches_reviewed', 'dependencies_reviewed']):
        raise ValueError('Missing full execution-path review for strict Triton')
    if result['issues']:
        raise ValueError('Candidate has unresolved static risks: ' + str(result['issues']))
    result['admission'] = 'review_bound'
    return result
