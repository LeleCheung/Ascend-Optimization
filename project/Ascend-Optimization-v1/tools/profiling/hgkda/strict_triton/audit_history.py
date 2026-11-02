"""隔离旧混合实现并核对存储改写范围；保留全部历史源码与原始成绩。"""
import ast
from pathlib import Path
from common import BATCH,HERE,PREVIOUS,read,save,sha,load

EXPLICIT_NATIVE={'rad2deg_','log10_','cumulative_trapezoid','negative_','expm1_','log2_',
    'fix','round_out','hardswish_out','special_round_out','hardswish','fmod_tensor_'}


def main():
    prepare=load('strict_history_prepare',HERE/'prepare.py');previous=read(PREVIOUS/'delivery.json');index=read(BATCH/'candidate-index.json')
    rows=[]
    for row in previous['rows']:
        op=row['operator'];old=Path((row.get('selected_new') or {}).get('source',PREVIOUS/'starting-points'/(op+'.py')))
        text,origin,method=prepare.original_text(op,PREVIOUS/'starting-points'/(op+'.py'))
        calls=[]
        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self,node):
                if any('triton.jit' in ast.unparse(d) for d in node.decorator_list):return
                self.generic_visit(node)
            def visit_Call(self,node):
                if isinstance(node.func,ast.Attribute):
                    name=ast.unparse(node.func)
                    if node.func.attr in {'contiguous','clone','copy_','reshape','flatten','to'} or name in {'torch.zeros','torch.ones','torch.rand'}:calls.append(name)
                self.generic_visit(node)
        Visitor().visit(ast.parse(text))
        disallowed=op in EXPLICIT_NATIVE or bool(calls)
        status='历史非Triton实现隔离，不计当前目标' if disallowed else '未发现禁止设备路径，仍以本轮具体审查为准'
        if op=='transform_bias_rescale_qkv':status='C++仅宿主胶水，实际Triton设备计算；经本轮绑定审查保留'
        rows.append(dict(operator=op,historical_source=str(old),historical_source_sha256=sha(old),
            explicit_native_device_path=op in EXPLICIT_NATIVE,storage_calls_rewritten=sorted(set(calls)),
            allocation_only_rewrite=index[op].get('storage_rewrite',False) and not calls,
            historical_disallowed=disallowed,status=status,current_source=index[op]['source'],current_source_sha256=index[op]['source_sha256']))
    value=dict(status='reviewed',rows=rows,explicit_native=sum(r['explicit_native_device_path'] for r in rows),
        with_implicit_storage=sum(bool(r['storage_calls_rewritten']) for r in rows),
        allocation_only=sum(r['allocation_only_rewrite'] for r in rows),
        disallowed_historical=sum(r['historical_disallowed'] for r in rows),
        scope='基于已人工审查的12项外部设备路径及存储调用AST位置；潜在复制/填充改为Triton。旧源码和原始成绩不移动、不覆盖，不按backend标签推断。')
    save(BATCH/'historical-isolation.json',value)
    print('HISTORICAL_ISOLATION',value['disallowed_historical'],'STORAGE',value['with_implicit_storage'],'ALLOCATION_ONLY',value['allocation_only'])


if __name__=='__main__':main()
