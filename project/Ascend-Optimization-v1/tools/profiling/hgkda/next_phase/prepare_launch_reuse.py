"""将历史SELU已验证的启动方式复用于negative候选；仍须独立完整评测。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


class Specialize(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name in {'_selu_nomask','_selu_masked','_selu_strided'}:
            for index,statement in enumerate(node.body):
                if isinstance(statement,ast.If) and isinstance(statement.test,ast.Compare) and 'dtype' in ast.unparse(statement.test):
                    node.body[index]=ast.Assign(targets=[ast.Name(id='y',ctx=ast.Store())],
                        value=ast.UnaryOp(op=ast.USub(),operand=ast.Name(id='x',ctx=ast.Load())))
        return node


def prepare(project,output,bitwise=False,block_cap=32768):
    if block_cap<1024 or block_cap>32768 or block_cap&(block_cap-1):
        raise ValueError('连续分块上限须为1024至32768之间的2次幂')
    origin=project/'operators/campaigns/endurance-20261022/deliverables/selu_/8764018cfb2cab9444a34caf7a171f7eac7395656631698ce491e676557462f0/main.py'
    tree=ast.parse(origin.read_text(encoding='utf8'))
    code=ast.unparse(ast.fix_missing_locations(Specialize().visit(tree)))
    code=code.replace('_selu_','_negative_').replace('_BLOCK_STRIDED = 1024','_BLOCK_STRIDED = 256')
    code=code.replace('SELU in-place: minimal per-call host work, full stride support, dtype-sized UB blocks.',
        'In-place negative: Triton computation with the historical validated launcher pattern.')
    if bitwise:
        # 已测负号候选丢失负零；此处翻转IEEE符号位，具体编译原因另查IR。
        code=code.replace('    y = -x','''    if X.dtype.element_ty == tl.float32:
        y = (x.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
    else:
        y = (x.to(tl.uint16, bitcast=True) ^ 0x8000).to(X.dtype.element_ty, bitcast=True)''')
        code=code.replace('maximum = 16384 if dtype is torch.float16 else 8192','maximum = '+str(block_cap))
    code='# Copyright 2026 FlagOS Contributors\n# SPDX-License-Identifier: Apache-2.0\n'+code+'\n'
    ast.parse(code)
    output.mkdir(parents=True,exist_ok=True);source=output/'main.py'
    if source.exists():assert source.read_text(encoding='utf8')==code
    else:source.write_text(code,encoding='utf8')
    record=dict(status='unmeasured_candidate',operator='negative_',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        origin=str(origin),origin_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),
        change='复用历史SELU已验证编译启动入口；'+('IEEE符号位翻转和连续分块上限'+str(block_cap)+'；' if bitwise else '设备计算改为负号；')+'非连续分块256；每次查询当前stream和地址，不缓存数据',
        hypothesis='减少每次JIT分派开销；连续大输入增加每program处理量；性能必须重新测量',
        gates=['完整KGS','换数据和stream','非连续邻接存储','空输入/偏移/转置/高rank/正负零'])
    (output/'provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bitwise',action='store_true')
    parser.add_argument('--block-cap',type=int,default=32768);args=parser.parse_args()
    print(json.dumps(prepare(args.project,args.output,args.bitwise,args.block_cap),ensure_ascii=False))
