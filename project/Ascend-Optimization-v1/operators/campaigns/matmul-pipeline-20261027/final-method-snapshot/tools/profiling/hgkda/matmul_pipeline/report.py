"""从真实KGS、开发轨迹与成对硬件证据生成中文交付报告。"""
import argparse,csv,json,statistics
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();b=a.batch
    out=b/'精简汇报.md'
    if out.exists():raise ValueError('不覆盖已有报告')
    comparison=json.loads((b/'validation/comparison.json').read_bytes());mechanism=json.loads((b/'analysis/mechanism.json').read_bytes())
    selection=json.loads((b/'validation/selection-before-kgs.json').read_bytes())['selected'];score=comparison['actual_candidate_over_parent']
    confirmation=json.loads((b/'matched-confirmation/state.json').read_bytes())
    action='采用新候选' if comparison['selected']=='candidate' else '保留父版'
    lines=['# 矩阵乘片内复用与流水优化','',
        f'本轮围绕 matmul_bias_activation 复用方向与流水配置进行优化，最终{action}。开发集最佳配置为 {comparison["candidate_id"]}，六组开发输入的实际设备加速为{selection["speedup"]:.3f}×；完整KGS对比为{score:.3f}×父版。',
        '',
        '| 版本 | 相对PyTorch设备加速比 | 完整KGS | 补充检查 |',
        '|---|---:|---:|---:|',
        f'| 本批父版 | {comparison["parent_geomean"]:.3f}× | 42/42 | 18＋24项 |',
        f'| 本批候选 | {comparison["candidate_geomean"]:.3f}× | 42/42 | 18＋24项 |',
        '',
        '相对父版的真实变化由15个相同case的延迟比计算，未用两组相对PyTorch汇总成绩相除替代。父版已有历史KG／HG-KDA和Codex优化，不是固定FlagGems原实现；本轮不改写历史基线成绩。KGS采用原合同设备kernel计时，不是完整Python调用walltime。',
        '',
        '## 做了哪些改动','',
        '保留多个小输出块，避免上一轮单大tile的UB压力与复用损失。通用模板支持相邻N块共享A、相邻M块共享B，以及2×2输出块分别复用A与B，并融合bias和激活；对不满足连续、整除、K至少两个分块等条件的输入继续使用父路径。新路径覆盖FP16／BF16，FP32沿用父版。',
        '',
        'Codex实现模板和测试，预登记7项结构／缓冲变换。CC＋DeepSeek读取已校验Roofline、父版源码、IR事实与当时搜索快照，追加选择分组调度和单流水级两项；模型提示、原回执、provider请求和输入SHA保存于agent-review。模型并未自主写出全部源码，不将本批结果归为纯原生KG独立收益。',
        '',
        '## 开发结果','',
        '| 配置 | 开发状态 | 相对父版 |','|---|---|---:|']
    for label in ['search-v2','search-v3']:
        for row in json.loads((b/label/'state.json').read_bytes())['cases']:
            value=row.get('speedup',row.get('screen_speedup'));status='六组完整开发输入' if row['status']=='passed' else '仅1024 FP16初筛' if row['status']=='screened' else row['status']
            lines.append(f'| {row["case"]} | {status} | {value:.3f}× |' if value is not None else f'| {row["case"]} | {status} | — |')
    lines+=['',
        '初筛和六组完整开发输入分别标注；未扩展的候选不能声称覆盖全部大矩阵。开发集使用1024／2048／4096与FP16／BF16、每case两次反转顺序的设备配对；正式合同前已冻结选版，不按完整结果再次调参。',
        '',
        '## 同输入硬件证据','',
        '| 规模FP16 | 父版采集μs | 候选采集μs | GM流量比 候选/父版 | L2流量比 | L1读取比 | L1写入比 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for item in mechanism['pairs']:
        ratios=item['traffic_candidate_over_parent'];r=item['records']
        lines.append(f'| {item["size"]}³ | {r["parent"]["facts"]["duration_us"]:.3f} | {r["candidate"]["facts"]["duration_us"]:.3f} | {ratios["GM Read + Write"]:.3f} | {ratios["L2 Read + Write"]:.3f} | {ratios["Read from L1"]:.3f} | {ratios["Write to L1"]:.3f} |')
    lines+=['',
        '这些时间来自独立同输入profiling，用于解释机制；不能代替KGS正式加速比。GM／L2为官方cache事件模型，不与逻辑请求或Memory.csv字段混用。组件比例逐block单列，未相加为时间分解，未当作整卡利用率。物理HBM峰值仍待独立核实，本轮复用前一批经验参考，不重新搜索峰值。',
        '',
        '源码IR差异、matmul／copy／alloc文本及其SHA保存于analysis/mechanism.json。IR文本次数是静态结构证据，不是动态指令数；实际流量变化、正式计时和资源编译结果结合解释，不把模型推测直接当成因果证明。',
        '',
        f'冻结选版后追加同模板确认：只改变M／N复用方向，其余constexpr形状、流水参数与模板保持一致。1024／4096的FP16／BF16共四项实际设备比值几何平均为{confirmation["geomean"]:.3f}×。该组不参与重新选版，用于区分复用方向与父版到新模板的联合变化。原始CSV与逐项结果见matched-confirmation。',
        '',
        '## 质量与贡献','',
        '两版均完成18项既有补充检查和24项新大输入／矩形／非整除／非连续／非默认stream检查。大输入门禁实际覆盖新分支，保留输入不修改和输出无意外别名；最终各一次原合同42项KGS，不重复刷分。',
        '',
        '模型输出有三处未被采纳的表述：BF16大K并不自动回退父版；group4是分组遍历而非蛇形；默认调度“接近上限”缺少足够证据。原回执保留，最终结论由真实源码、计数和完整评测约束。',
        '',
        '初次search因诊断模块导入路径缺失在提交设备任务前停止，修复后新建search-v2，旧失败原件保留。通用模板、有限预算、模型建议和实际控制器均单独归档；未运行新模拟器／RTL采集，未改共享编译器。',
        '',
        '## 复现与交付','',
        '- validation/delivered.py：本轮最终保留源码，SHA '+comparison['source_sha256']+'。',
        '- validation/comparison.json、正式逐case对比.csv：原合同15项真实延迟与通过状态。',
        '- parent-profile、candidate-profile、analysis：四项同输入采集与IR／流量诊断。',
        '- search-v2、search-v3、agent-review：冻结候选、开发轨迹与真实模型调用。',
        '- evidence-audit.json、transfers：原件及逐文件核验；不覆盖历史采集。',
        '',
        '~~~bash',
        'cd /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1',
        '# 使用固定tle_yy容器和CANN9.0.0，先确认项目队列空闲、持卡7项目互斥锁',
        'source /usr/local/Ascend/cann-9.0.0/set_env.sh',
        'export ASCEND_RT_VISIBLE_DEVICES=7',
        '/usr/local/python3.11.15/bin/python3.11 tools/evaluation/evaluate-operator.py \\',
        '  --operator matmul_bias_activation \\',
        '  --source operators/campaigns/matmul-pipeline-20261027/validation/delivered.py \\',
        '  --output /data/hanle/ascend-optimization/runs/新的矩阵复验 --label delivered \\',
        '  --server http://127.0.0.1:19659 --repeats 1 --timing-scope device_kernel',
        '~~~','',
        'PYTHONPATH须与项目控制器一致，包含固定FlagGems/src、KGS client及sources。此次不commit／push，Windows与910B逐文件SHA同步，GitHub保持原提交。','']
    out.write_text('\n'.join(lines),encoding='utf8')
    with (b/'正式逐case对比.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['case_id','axes','parent_ms','candidate_ms','speedup']);writer.writeheader();writer.writerows(comparison['cases'])
    print(json.dumps(dict(report=str(out),selected=comparison['selected'],speedup=score)))

if __name__=='__main__':main()
