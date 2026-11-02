"""恢复原Triton源码并替换隐式PyTorch搬运；固定合同与历史来源。"""
import ast
import shutil
from pathlib import Path
from common import BATCH,HERE,PREVIOUS,read,save,sha
from gate import inspect


class StorageRewrite(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if any('triton.jit' in ast.unparse(d) for d in node.decorator_list):return node
        return self.generic_visit(node)
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if not isinstance(node.func,ast.Attribute):return node
        name=ast.unparse(node.func);attr=node.func.attr
        if name in {'torch.zeros','torch.ones','torch.rand','torch.empty_like'}:
            node.func=ast.Name(id={'torch.zeros':'_strict_zeros','torch.ones':'_strict_ones','torch.rand':'_strict_rand','torch.empty_like':'_strict_empty_like'}[name],ctx=ast.Load())
        elif attr in {'contiguous','clone','copy_','reshape','flatten','to'}:
            helper={'contiguous':'_strict_contiguous','clone':'_strict_clone','copy_':'_strict_copy_into','reshape':'_strict_reshape','flatten':'_strict_flatten','to':'_strict_cast'}[attr]
            node.args=[node.func.value]+node.args;node.func=ast.Name(id=helper,ctx=ast.Load())
        return node


def original_text(op,current):
    text=current.read_text(encoding='utf8');method='原字节保留'
    if op=='log10_':
        text=text.split('# 计算来自本项目Ascend C源码',1)[0]+'\nrun=_triton_run\n';method='移除Ascend C桥，只保留原_triton_run'
    elif op in {'rad2deg_','negative_','expm1_','log2_','fix','round_out','special_round_out','hardswish','fmod_tensor_'}:
        text=text.split('import hashlib as _hashlib',1)[0].split('_original_run=run',1)[0]
        method='移除追加Ascend C桥，保留此前run'
    elif op=='cumulative_trapezoid':
        current=PREVIOUS.parent/'reference-closure-20261031/proposals/cumulative_trapezoid/tiled-restore-v2/main.py'
        text=current.read_text(encoding='utf8');method='恢复CANN扫描接入前的纯Triton转置／扫描／还原版本'
    elif op=='hardswish_out':
        current=PREVIOUS/'proposals/hardswish_out/ascendc-large-v3/main.py'
        text=current.read_text(encoding='utf8').split('"""FlagGems hardswish math',1)[1]
        text='"""FlagGems hardswish math'+text
        text=text.rsplit('def run(x,out):',1)[0]+'def run(x,out):\n    return _out_fallback(x,out)\n'
        method='提取out版Triton计算与布局路径，移除外部Ascend C调用'
    elif op=='transform_bias_rescale_qkv':
        # 恢复独立Triton宿主入口，减少首轮依赖范围；已核验C++宿主桥可后续独立恢复。
        text=(PREVIOUS/'starting-points'/('transform_bias_rescale_qkv.py')).read_text(encoding='utf8')
    return text,current,method


def main():
    if (BATCH/'plan.json').exists():raise FileExistsError('独立批次已准备，不覆盖冻结源码')
    BATCH.mkdir(parents=True,exist_ok=True)
    history=read(PREVIOUS/'delivery.json');old_plan=read(PREVIOUS/'plan.json');indices={r['operator']:r for r in old_plan['records']}
    helper=(HERE/'triton_storage.py').read_text(encoding='utf8')
    records=[];reviews=[];excluded=[]
    for row in history['rows']:
        op=row['operator'];current=PREVIOUS/'starting-points'/(op+'.py')
        chosen=row.get('selected_new') or {}
        history_source=Path(chosen.get('source',current))
        previous_inspect=inspect(history_source)
        excluded.append(dict(operator=op,source=str(history_source),source_sha256=sha(history_source),
            preliminary=previous_inspect,scope='所有历史选中版本均需重新审核，不按旧backend标签判定'))
        text,origin,method=original_text(op,current)
        original_tree=ast.parse(text);tree=StorageRewrite().visit(ast.parse(text));ast.fix_missing_locations(tree)
        changed=ast.dump(tree)!=ast.dump(original_tree)
        if changed:text=ast.unparse(tree)+'\n\n'+helper
        source=BATCH/'candidates'/op/'restored-v1/main.py';source.parent.mkdir(parents=True,exist_ok=True)
        source.write_bytes(text.encode('utf8'))
        inspected=inspect(source);save(source.parent/'static-audit.json',inspected)
        record=dict(indices[op]);leaf=BATCH/'catalog/ops'/op;leaf.mkdir(parents=True,exist_ok=True)
        hashes={}
        for name in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']:
            p=leaf/name;p.write_bytes((PREVIOUS/record['catalog_leaf']/name).read_bytes());hashes[name]=sha(p)
        record.update(seed=source.relative_to(BATCH).as_posix(),source_sha256=sha(source),catalog_leaf=leaf.relative_to(BATCH).as_posix(),contract_sha256=hashes)
        records.append(record)
        review=dict(operator=op,source=str(source),source_sha256=sha(source),status='pending_manual_review',
            all_branches_reviewed=False,dependencies_reviewed=False,origin=str(origin),origin_sha256=sha(origin),
            method=method,storage_rewrite=changed,helper_sha256=sha(HERE/'triton_storage.py') if changed else None,
            attribution='历史KG/CC/DeepSeek与Codex Triton候选恢复；Codex移除非Triton路径并替换隐式PyTorch搬运，不计新性能收益')
        save(source.parent/'review.json',review);reviews.append(review)
        print(op,'storage_rewrite',changed,'STATIC',inspected['static_passed'],inspected['issues'])
    save(BATCH/'plan.json',dict(old_plan,records=records,schema='hgkda-strict-triton-v1',measurement_policy='严格Triton先审核全部路径，再按原KGS输入/容差/参数正式复测；参考资格复用已核验原件，未知请求不重发'))
    save(BATCH/'catalog/manifest.json',dict(api_version='v6.2',name=BATCH.name,evaluator='native',layout='per-operator'))
    save(BATCH/'historical-backend-audit.json',dict(rows=excluded,scope='旧25/30不沿用为严格Triton计数'))
    save(BATCH/'candidate-index.json',{r['operator']:r for r in reviews})
    for name in ['software-lock.json']:(BATCH/name).write_bytes((PREVIOUS/name).read_bytes())


if __name__=='__main__':main()
