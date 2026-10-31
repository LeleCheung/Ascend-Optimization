"""910B宿主：一条命令对当前交付源码做新的完整KGS评测和补测。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',default='threeway-20261030')
    parser.add_argument('--operator',required=True)
    parser.add_argument('--run-name',required=True,help='新结果目录名，例如manual-20261031')
    args=parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',args.run_name):
        raise ValueError('run-name只允许小写字母、数字和连字符')
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(here))
    spec=importlib.util.spec_from_file_location('threeway_reproduction_runner',here/'run.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    with (controller.base.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        runner=controller.Runner(args.batch)
        if args.operator not in runner.q.records:raise ValueError('算子未在本批登记')
        best=runner.state['operators'][args.operator].get('best') or {}
        if controller.base.extra_qualification(best)!='passed':raise ValueError('当前源码尚未通过补测')
        source=Path(best['source'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=best['source_sha256']:
            raise ValueError('源码SHA与交付记录不同')
        out=runner.batch/'reproductions'/args.run_name/args.operator
        if out.exists():raise FileExistsError('使用新run-name；不覆盖旧结果 '+str(out))
        runner.wait_idle()
        # evaluate按manifest选择原生Python/Ascend C或Triton合同入口，计时参数保持相同。
        result=runner.evaluate(args.operator,source,out/'formal','HG-KDA-reproduction')
        if result.get('status')=='PASSED':
            gates=out/'extra-checks.json'
            result.update(extra_checks=str(gates),supplementary_status=runner.supplementary(
                args.operator,source,gates,'reproduction-'+args.operator+'-'+args.run_name))
            if args.operator=='cross_out':
                safety=out/'cross-safety.json'
                runner.q.run('reproduction-cross-safety-'+args.run_name,controller.base.quote([
                    controller.base.PY,here/'check_cross.py','--source',source,'--output',safety]),1200)
                value=json.loads(safety.read_text(encoding='utf8'))
                result['cross_safety']=str(safety)
                if value.get('status')!='passed':result['supplementary_status']='cross_safety_failed'
        result.update(operator=args.operator,scope='一次新完整KGS加独立补测；不替换历史最佳，不计为新优化')
        controller.base.save_json(out/'reproduction.json',result)
        print(json.dumps({k:result.get(k) for k in ['operator','status','geo_mean','num_passed','num_workloads','supplementary_status','source_sha256','result']},ensure_ascii=False))
        runner.wait_idle()
        if result.get('status')!='PASSED' or result.get('supplementary_status')!='passed':
            raise SystemExit('复现未通过，原始失败保留在新目录')


if __name__=='__main__':main()
