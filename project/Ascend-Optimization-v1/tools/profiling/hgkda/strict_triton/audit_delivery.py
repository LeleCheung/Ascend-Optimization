"""交付审计：30项源码与原合同、KGS请求绑定、case集合及编译证据逐项核验。"""
import io
import ast
import json
from pathlib import Path
import sys
from common import BATCH,HERE,PROJECT,PREVIOUS,read,save,sha,load
from gate import admit


def main():
    sys.path.insert(0,str(PROJECT.parents[1]/'sources/kernelgen_server/client'))
    from kernelgen_client.operator_bundles import pack_operator_bundle
    load('strict_final_report',HERE/'report.py').main()
    plan=read(BATCH/'plan.json');indices=read(BATCH/'candidate-index.json');delivery=read(BATCH/'delivery.json')
    records={r['operator']:r for r in plan['records']};checks=[];identities={}
    assert len(records)==30 and len(delivery['rows'])==30
    assert sum(r['group']=='1015' for r in delivery['rows'])==10 and sum(r['group']=='1029' for r in delivery['rows'])==20
    for row in delivery['rows']:
        op=row['operator'];record=records[op];source=Path(row['source']);admit(source,source.parent/'review.json')
        leaf=BATCH/record['catalog_leaf'];old=PREVIOUS/record['catalog_leaf']
        for name,digest in record['contract_sha256'].items():
            assert sha(leaf/name)==digest
            identities[str(leaf/name)]=digest
        old_record=next(r for r in read(PREVIOUS/'plan.json')['records'] if r['operator']==op)
        old=PREVIOUS/old_record['catalog_leaf']
        for name in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']:
            assert (leaf/name).read_bytes()==(old/name).read_bytes(),(op,name,'原合同不同')
        compared=[]
        for side in ['pytorch','cann']:
            measurement=row.get(side) or {}
            if not measurement.get('path'):continue
            target_leaf=leaf if side=='pytorch' else BATCH/'cann-catalog/ops'/op
            catalog_note='本批冻结catalog'
            if side=='cann' and measurement.get('measurement')=='reused_exact_source_and_contract':
                # 复用必须核对真正提交过的历史bundle。新包装只少一行注释，AST相同但tar哈希不同。
                reference=next(r for r in read(PREVIOUS/'delivery.json')['rows'] if r['operator']==op)
                historical_leaf=PREVIOUS/'cann-catalog'/reference['reference_source_sha256'][:12]/'ops'/op
                for name in ['definition.json','correctness.jsonl','timing.jsonl']:
                    assert (historical_leaf/name).read_bytes()==(target_leaf/name).read_bytes()
                assert ast.dump(ast.parse((historical_leaf/'oracle.py').read_text()))==ast.dump(ast.parse((target_leaf/'oracle.py').read_text()))
                target_leaf=historical_leaf
                catalog_note='复用历史实际提交bundle；本批CANN包装仅注释不同，输入与参考AST相同'
            archive=io.BytesIO();digest,_=pack_operator_bundle(target_leaf,archive)
            result_path=Path(measurement['path']);request=result_path.with_name(result_path.name.replace('.result.json','.request.json'))
            payload=read(request);assert payload['binding']==dict(bundle_id='sha256:'+digest,definition=op)
            assert payload['implementation']['sources'][0]['content']==source.read_text(encoding='utf8')
            value=read(result_path)
            expected={json.loads(line)['name'] for name in ['correctness.jsonl','timing.jsonl'] for line in (target_leaf/name).read_text().splitlines() if line.strip()}
            observed={v['uuid'] for v in value.get('per_workload',[])}
            assert value['num_workloads']==len(expected)
            assert observed==expected,(op,side,'case集合不一致',expected-observed,observed-expected)
            for file in [request,result_path]:identities[str(file)]=sha(file)
            compared.append(dict(side=side,bundle_id='sha256:'+digest,cases=len(expected),measurement=measurement.get('measurement'),
                catalog_leaf=str(target_leaf),catalog_note=catalog_note,result=str(result_path)))
        checks.append(dict(operator=op,source_sha256=sha(source),backend_review='passed',original_contract='identical',
            comparisons=compared,accepted=row['accepted'],status=row['status'],compiled_path=row['compiled_path']))
    output=BATCH/'delivery-audit.json'
    save(output,dict(status='verified',operators=30,groups=delivery['groups'],accepted=delivery['accepted'],
        delivery_sha256=sha(BATCH/'delivery.json'),checks=checks,files_sha256=identities,
        scope='核验全部30项身份及已发生请求；未达标与参考缺口保留，不以审计通过表示30项均通过性能验收'))
    print('DELIVERY_AUDIT_VERIFIED',len(checks),'ACCEPTED',delivery['accepted'])


if __name__=='__main__':main()
