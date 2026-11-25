"""Numerical references from the same GEMM capture; empirical envelopes and nominal bounds are separate."""
import math

def derive(facts,shape,dtype,empirical):
    if len(shape)!=3 or any(type(v) is not int or v<=0 for v in shape):raise ValueError('GEMM requires positive integer M, N, and K')
    if dtype not in ['float16','bfloat16','float32']:raise ValueError('Missing workload contract for this precision')
    flops=2*math.prod(shape);duration=facts['duration_us']
    if type(duration) not in [int,float] or not math.isfinite(duration) or duration<=0:raise ValueError('Invalid same-capture duration')
    compute=[r['tflops'] for r in empirical['records'] if r['kind']=='gemm' and r['dtype']==dtype]
    copy=[empirical['gm_counter_reference']['gbps']] if empirical.get('gm_counter_reference') else []
    if not compute or not copy:raise ValueError('Missing same-precision compute or same-counter-scope empirical GM reference; logical copy bytes cannot be substituted')
    if any(type(v) not in [int,float] or not math.isfinite(v) or v<=0 for v in compute+copy):raise ValueError('Empirical reference contains invalid values')
    compute=max(compute);bandwidth=max(copy)
    curves=[]
    for c in facts['curves']:
        if c['official_bytes'] is None or c['official_bytes']<=0:continue
        # Keep component-local work for Vector points; do not assign the entire GEMM 2MNK to every pipeline.
        curve=dict(c);curve['work_scope']='Official component-local operation count, not whole-GEMM work'
        if 'Cube' in c['component'] or c['path'] in ['GM Read + Write','L2 Read + Write']:
            curve.update(algorithmic_flops=flops,algorithmic_tflops=flops/duration/1e6,
                hardware_over_algorithmic=c['official_flops']/flops,algorithmic_ai=flops/c['official_bytes'],
                work_scope='Report effective GEMM 2MNK separately from official internal operation counts; fused epilogue FLOPs are excluded')
        if c['path'] in ['GM Read + Write','L2 Read + Write']:
            bw=bandwidth if c['path']=='GM Read + Write' else c['bandwidth_tbps']*1000
            compute_roof=compute
            bound=min(compute_roof,curve['algorithmic_ai']*bw/1000)
            curve.update(reference_compute_tflops=compute_roof,reference_bandwidth_gbps=bw,
                reference_bound_tflops=bound,reference_achievement=curve['algorithmic_tflops']/bound,
                limiting_side='memory' if curve['algorithmic_ai']*bw/1000<compute_roof else 'compute',
                reference_kind='Empirical observation envelope, not a proven physical bound' if c['path']=='GM Read + Write' else 'Official L2 bandwidth model plus same-precision empirical effective-compute reference')
        curves.append(curve)
    issues=[]
    if not curves:issues.append('No valid official Roofline data; refusing a plot based on zero counts')
    if any(c.get('reference_achievement',0)>1.05 for c in curves):issues.append('Above reference line; check counts, duration, and limits without clipping values')
    if any(c['path']=='GM Read + Write' and c['bandwidth_tbps']*1000<bandwidth*.95 for c in curves):
        issues.append('Official GM constant is below persistent-copy observations under the same PMU model; retain it without claiming a calibrated physical bound for this device')
    return dict(algorithmic_flops=flops,algorithmic_tflops=flops/duration/1e6,duration_us=duration,
        curves=curves,issues=issues,model_scope='Official PMU-derived points from the same capture; report empirical GM envelopes separately from nominal models',
        diagnosis='Compare GM and L2 gaps first; a point far below an envelope does not establish bandwidth limitation. Inspect pipeline/resource pressure with controlled tiling, waits, and IR.')
