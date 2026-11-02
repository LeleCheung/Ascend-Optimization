"""严格Triton后端准入：AST发现禁用路径＋绑定人工审查清单，非完整静态证明。"""
import ast
from pathlib import Path
from common import read,save,sha

FORBIDDEN_METHODS={'contiguous','clone','copy_','fill_','zero_','flatten','reshape','to','cpu','cuda','numpy','resolve_conj','resolve_neg'}
FORBIDDEN_TORCH={'zeros','ones','full','rand','randn','arange','tensor','as_tensor','from_numpy','empty_strided'}

def inspect(source):
    source=Path(source);text=source.read_text(encoding='utf8');tree=ast.parse(text)
    issues=[];kernels=[];imports=[];host_calls=set();external=[]
    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self,node):
            if any('triton.jit' in ast.unparse(d) for d in node.decorator_list):
                kernels.append(node.name)
                for n in ast.walk(node):
                    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func).startswith('torch.'):
                        issues.append(dict(line=n.lineno,reason='Triton核内出现torch计算',call=ast.unparse(n.func)))
                return
            self.generic_visit(node)
        def visit_Import(self,node):
            imports.extend(a.name for a in node.names);self.generic_visit(node)
        def visit_ImportFrom(self,node):
            imports.append(node.module or '');self.generic_visit(node)
        def visit_Call(self,node):
            name=ast.unparse(node.func);host_calls.add(name)
            if isinstance(node.func,ast.Attribute):
                attr=node.func.attr
                if attr in FORBIDDEN_METHODS or name.startswith('torch.ops.') or name in {'torch.'+x for x in FORBIDDEN_TORCH}:
                    issues.append(dict(line=node.lineno,reason='宿主计算／搬运必须改为Triton',call=name))
                if name in {'torch.exp','torch.log','torch.cumsum','torch.neg','torch.add','torch.mul','torch.minimum','torch.maximum'}:
                    issues.append(dict(line=node.lineno,reason='目标数值操作不允许由torch执行',call=name))
            if name in {'eval','exec','__import__'}:issues.append(dict(line=node.lineno,reason='动态执行须拒绝',call=name))
            self.generic_visit(node)
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value.endswith('.so'):external.append(node.value)
    Visitor().visit(tree)
    # 不按注释或源码中的历史术语拦截；这里只识别实际导入／调用和二进制加载。
    for name in imports:
        if name in {'ctypes','subprocess'}:issues.append(dict(reason='外部执行入口须审查',module=name))
    if external:issues.append(dict(reason='外部二进制只接受已人工核验的Triton宿主桥',paths=external))
    return dict(schema='hgkda-strict-triton-static-v1',source=str(source),source_sha256=sha(source),
        static_passed=not issues,issues=issues,triton_kernels=kernels,imports=sorted(set(imports)),
        host_calls=sorted(host_calls),external_binaries=external,
        scope='AST扫描仅识别已知禁用入口；必须与绑定源码和依赖的人工逐路径审查及运行证据共同使用')

def admit(source,review_path):
    result=inspect(source);review=read(review_path)
    assert review['source_sha256']==result['source_sha256'] and review['status']=='reviewed_strict_triton'
    assert review['all_branches_reviewed'] and review['dependencies_reviewed']
    issues=list(result['issues'])
    bridges=review.get('reviewed_host_bridges',[])
    if bridges:
        assert {Path(x).resolve() for x in result['external_binaries']}=={Path(b['binary']).resolve() for b in bridges}
        for bridge in bridges:
            assert bridge['candidate_sha256']==result['source_sha256']
            assert bridge['status']=='reviewed_host_only' and bridge['all_functions_reviewed']
            assert bridge['device_computation']=='Triton_only'
            assert sha(bridge['binary'])==bridge['binary_sha256']
            assert sha(bridge['cpp_source'])==bridge['cpp_source_sha256']
            if bridge.get('binary_snapshot'):assert sha(bridge['binary_snapshot'])==bridge['binary_sha256']
        # 只豁免已绑定、逐函数审查的宿主启动桥；ctypes/厂商目标调用等其余问题仍拒绝。
        issues=[v for v in issues if v.get('reason')!='外部二进制只接受已人工核验的Triton宿主桥']
    assert not issues,issues
    result['admitted']=True;result['host_bridges']=bridges
    return result
