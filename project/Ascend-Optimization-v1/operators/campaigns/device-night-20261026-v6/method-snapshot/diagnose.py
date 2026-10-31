"""从受控设备干预产生可检验机制假设与下一次采集；不把活跃比例当作因果。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def positive(values):
    if not values or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('无效设备计时')
    return statistics.median(values)


def diagnose(probe):
    if probe['schema']!='hgkda-device-interventions-v1':raise ValueError('证据schema不同')
    op=probe['operator'];cases=[];rank={}
    for case in probe['cases']:
        passed={v['name']:v for v in case['variants'] if v['status']=='passed'}
        control=passed.get('reexpress_parent');observations=[]
        if control is None:
            cases.append(dict(id=case['id'],status='no_control',action='retain'));continue
        ctimes=control['timings']['variant_ms'];positive(ctimes)
        for name,item in passed.items():
            times=item['timings']['variant_ms'];positive(times)
            changes={key:dict(before=control['config'][key],after=value) for key,value in item['config'].items() if value!=control['config'][key]}
            ratio=statistics.median(ctimes)/statistics.median(times)
            separated=min(ctimes)/max(times)>1.05
            # 每项单参数/关联tile变换的机制解释预登记；证据仍需后续采集确认。
            if op=='matmul_bias_activation':
                hypotheses={
                    'compact_m':('tile_resource_pressure','小M块降低单任务缓冲压力，但增加B重复请求；检查UB/存储和设备延迟'),
                    'compact_n':('tile_resource_pressure','小N块减少双累加器占用，但增加A重复请求；检查缓冲与搬运'),
                    'larger_k':('loop_transfer_granularity','大K块减少循环/搬运指令次数，同时增加缓冲；检查IR和等待'),
                    'group4':('cache_reuse','分组调度不改变逻辑请求量，可能改善局部复用；检查缓存命中和实际搬运'),
                    'no_multibuffer':('pipeline_or_resource_pressure','关闭多缓冲干预流水/缓冲，不等同证明重叠不足；检查等待、缓冲和Cube活跃')}
                m,n,k=case['spec']['shape'];cfg=item['config'];byte=4 if case['spec']['dtype']=='float32' else 2
                static=dict(programs=(m//cfg['bm'])*(n//(2*cfg['bn'])),
                    requested_A_bytes=m*k*(n//(2*cfg['bn']))*byte,
                    requested_B_bytes=k*n*(m//cfg['bm'])*byte,
                    k_loop_iterations_per_program=k//cfg['bk32' if byte==4 else 'bk16'],
                    scope='规则整除形状的源码逻辑请求估计；不是实测HBM字节')
            else:
                hypotheses={
                    'wider_transfer':('transfer_granularity','扩大连续列并缩短归约片，检查搬运粒度与循环/等待'),
                    'more_rows':('program_organization','一次归约更多行，检查program数、归约指令与缓冲'),
                    'no_multibuffer':('pipeline_or_resource_pressure','多缓冲可能加大资源压力；需结合编译缓冲和等待验证'),
                    'fp32_accumulator':('precision_expansion','比较fp16与fp32中间值；BF16和FP32不受此开关影响'),
                    'short_reduction_tile':('reduction_resource_pressure','缩短归约片降低片上占用，但增加循环；检查缓冲和指令')}
                static=dict(scope='归约中间精度与分块来自配置；实际cast及存储须查本case IR',
                            fp16_native=bool(item['config']['native'] and case['spec']['dtype']=='float16'))
            hypothesis,reason=hypotheses.get(name,('configuration_control','相同调度配置的重表达对照'))
            observations.append(dict(variant=name,changed=changes,hypothesis=hypothesis,reason=reason,
                speedup_over_reexpressed_control=ratio,sample_ranges_separated=separated,
                status='direction_supported_pending_confirmation' if separated else 'not_established',static=static,
                limits='对照与变体为同输入独立交错父版测量；变体间不是同时配对，需胜出候选再次配对确认'))
            rank.setdefault(name,[]).append(ratio)
        cases.append(dict(id=case['id'],spec=case['spec'],status='analyzed',observations=observations,
            next_probe=dict(kind='paired_final_confirmation',reason='先确认候选与父版同输入配对设备收益，再采集指标解释变化')))
    ranked=sorted([dict(variant=k,geomean=statistics.geometric_mean(v),cases=len(v)) for k,v in rank.items()],key=lambda r:r['geomean'],reverse=True)
    return dict(schema='hgkda-device-diagnosis-v1',operator=op,source_sha256=probe['source_sha256'],cases=cases,ranking=ranked,
        acquisition_policy=['真机延迟先验证效应；OpProf检查资源与搬运变化','IR检查cast、缓冲、搬运、同步和Cube指令的代码机制',
                            '模拟器仅对小诊断case、core0、有限预算采集；不把模拟器时长当真机时长'],
        unknown=['尚无设备时间关键路径或完整指令依赖图','本轮没有定量Roofline和RTL执行器'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('不覆盖已有诊断')
    result=diagnose(json.loads(a.probe.read_text()));result['probe_sha256']=hashlib.sha256(a.probe.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
