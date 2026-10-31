"""为历史amin起点补齐PyTorch的NaN传播；两组共享，不计为搜索收益。"""
import ast


def amin_nan(source):
    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name) or node.func.value.id != 'tl':
                return node
            if node.func.attr == 'minimum':
                node.keywords.append(ast.keyword(arg='propagate_nan',value=ast.parse('tl.PropagateNan.ALL',mode='eval').body))
            elif node.func.attr == 'min':
                node.func.attr = 'reduce'
                if len(node.args) == 1 and not any(k.arg == 'axis' for k in node.keywords):
                    node.keywords.append(ast.keyword(arg='axis',value=ast.Constant(None)))
                node.keywords.append(ast.keyword(arg='combine_fn',value=ast.Name(id='_hg_nan_min',ctx=ast.Load())))
            return node
    tree=Rewrite().visit(ast.parse(source))
    helper=ast.parse('@triton.jit\ndef _hg_nan_min(a,b):\n    return tl.minimum(a,b,propagate_nan=tl.PropagateNan.ALL)\n').body[0]
    index=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.FunctionDef))
    tree.body.insert(index,helper);ast.fix_missing_locations(tree)
    output=ast.unparse(tree)+'\n';compile(output,'amin_nan.py','exec');return output
