"""证据约束的调用执行图和竞争机制诊断；不推断未观测的设备依赖。"""
from __future__ import annotations
import hashlib
import json
import math
import statistics

VERSION = 'launch-mechanism-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def interval(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('计时样本缺失或无效')
    ordered = sorted(values)
    # 展示原始重复观测的范围，不将此范围解释为统计置信区间。
    return dict(median=statistics.median(values), minimum=ordered[0], maximum=ordered[-1], n=len(values))


def diagnose(probe):
    if probe.get('schema') != 'hgkda-execution-slice-v1':
        raise ValueError('执行切片 schema 不同')
    graphs, decisions = [], []
    for case in probe['cases']:
        if case['status'] != 'passed':
            decisions.append(dict(case=case['id'], action='retain', reason='证据采集失败', error=case.get('error')))
            continue
        obs = case['timings']
        wrapper, direct = interval(obs['wrapper_us']), interval(obs['direct_us'])
        paired = [w/d for w,d in zip(obs['wrapper_us'],obs['direct_us'])]
        if len(paired) != len(obs['wrapper_us']) or len(paired) != len(obs['direct_us']):
            raise ValueError('配对计时数量不同')
        # 实质改进阈值在开发前登记，避免把微小噪声作为机制证据。
        dispatch = min(paired) > 1.15
        graph = dict(case=case['id'], source_sha256=probe['source_sha256'],
            scope='warm_cache_single_stream_full_call',
            nodes=[dict(id='wrapper',kind='python_wrapper'), dict(id='jit',kind='jit_dispatch'),
                   dict(id='submit',kind='runtime_submission'),dict(id='device',kind='device_execution'),
                   dict(id='completion',kind='synchronization')],
            edges=[dict(source=a,target=b,basis='捕获到的JIT调用与同stream完成语义') for a,b in
                   [('wrapper','jit'),('jit','submit'),('submit','device'),('device','completion')]],
            observations=dict(wrapper_us=wrapper,direct_us=direct,paired_ratio=interval(paired),
                              launches=case['launches']),
            unknown=['未测量每条边的独占耗时','不含设备内部指令依赖','不从同步walltime推导纯设备耗时'])
        graphs.append(graph)
        tiles=[]
        for item in case.get('tile_interventions',[]):
            if item['status']!='passed':continue
            parent=item['timings']['parent'];variant=item['timings']['variant']
            if len(parent)!=len(variant):raise ValueError('分块配对数量不同')
            ratios=interval([a/b for a,b in zip(parent,variant)])
            tiles.append(dict(block=item['block'],grid=item['grid'],paired_ratio=ratios,
                supported=ratios['minimum']>1.1,
                scope='同JIT分派的完整任务组织差异，未单独辨识设备耗时'))
        supported_tiles=[x for x in tiles if x['supported']]
        best_tile=max(supported_tiles,key=lambda x:x['paired_ratio']['median']) if supported_tiles else None
        decisions.append(dict(case=case['id'],
            competing_hypotheses=[dict(id='dispatch',evidence='相同编译产物直接提交的配对实验',
                                       supported=dispatch),
                                  dict(id='device_or_other',evidence='直接提交未改变设备程序；剩余耗时尚需分块探针',
                                       supported=None)],
            action=('cache_dispatch_and_change_tile' if dispatch and best_tile else
                    'cache_compiled_dispatch' if dispatch else 'change_tile' if best_tile else 'retain_or_probe'),
            tile_evidence=tiles,recommended_tile=best_tile,
            prediction=dict(metric='same_case_wrapper_latency',direction='decrease' if dispatch else 'unknown',
                            mechanism='减少Python/JIT分派；设备程序和grid保持不变' if dispatch else '需继续区分设备组织与其他成本'),
            limits='直接提交是干预探针，候选必须对动态指针、当前stream、布局及数值完整复验'))
    return dict(schema='hgkda-mechanism-diagnosis-v1',policy=VERSION,
                source_sha256=probe['source_sha256'],graphs=graphs,decisions=decisions)
