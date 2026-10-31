"""从实际评测与数值诊断生成本轮中文报告和逐case对比，不修改历史。"""
import argparse,csv,hashlib,json,math
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);a=p.parse_args()
    b=a.batch;out=b/'最终报告.md'
    if out.exists():raise ValueError('不覆盖已有报告')
    comparison=json.loads((b/'validation/comparison.json').read_bytes());analysis=json.loads(a.analysis.read_bytes())
    data={r['id'].split('/')[-1]:r for r in analysis['cases']}
    lines=['# 910B 数值 Roofline 接入与矩阵乘验证','',
        '## 可汇报结论','',
        '复用了导师提供的Roofline分析代码，新增910B计数与时间校验、分组件数值对照、中文图表以及KG角色摘要接入。历史128倍差异已经定位：CSV固定按每条Cube矩阵指令8192次运算估计，而当前宏指令工作量随tile变化，三个受控实验差异分别为32、128、256倍，不能统一乘128。',
        '',f'矩阵乘父版与候选都通过42/42完整KGS和18/18补充检查；本批父版相对PyTorch为{comparison["parent_geomean"]:.3f}×，候选为{comparison["candidate_geomean"]:.3f}×。按同case实际延迟计算，候选相对父版仅{comparison["actual_candidate_over_parent"]:.3f}×，因此保留父版，本轮没有新增算子性能收益。',
        '',
        '## 数值口径与校准','',
        '| 项目 | 本轮结果 |','|---|---|',
        '| 导师分析器 | 原逻辑提取，保留Markdown来源与代码SHA |',
        '| 有效矩阵工作量 | 2MNK，不包含bias／激活尾处理 |',
        '| CSV Cube工作量 | 固定8192乘指令数；不能统一修正倍数 |',
        '| 官方Roofline工作量 | 来自另一组事件；FP16／BF16受控实验匹配2MNK，FP32内部操作约4倍 |',
        '| GM／L2流量 | 官方cache事件派生；与Memory.csv接口流量、逻辑字节分别报告 |',
        '| Vector组件 | 仅报告本组件内部操作，不能重复分配整个2MNK |',
        '| GM经验参考 | 同PMU模型persistent copy字节／同次设备时间；不是硬件峰值 |',
        '| 计算经验参考 | FP16、BF16、FP32分别测量；预分配输出 |',
        '| 纯copy | 单列吞吐，不构造有意义的GEMM FLOPs点 |',
        '',
        '同GM事件模型的persistent copy观测约1.148TB/s，已高于官方0.8TB/s模型。大数组torch copy的有效逻辑吞吐约1.250TB/s，另列、不代入GM计数字节的参考线。模型匹配仍未解决，因此当前交付是有来源的数值点与经验／官方参考对照，尚不是完全标定的物理HBM上限。',
        '',
        '固定环境为FlagGems d6a8eec473517a3d68157b208eb9c057eb1d4c50、FlagTree 0.6.0+ascend.gitc286cba6、CANN9.0.0、Python3.11.15、tle_yy容器、物理卡7。本轮保存msprof与编译／计时后端字节SHA。官方源码研究快照为a63063569742c1ba7be9688fc7799a8248574dee；尚未证明它与安装二进制逐源对应，也未独立读取Hal GmType。msprof op --version在本机不支持，该错误原样留档，版本依据组件version.info和二进制SHA。',
        '',
        '## 同输入的数值证据','',
        '| 精度／规模 | 父版设备采集时间μs | 候选设备采集时间μs | PyTorch GEMM组件μs | 父版GM经验参考达成率 |',
        '|---|---:|---:|---:|---:|']
    for dt in ['float16','float32']:
        for size in [1024,2048]:
            parent=data['parent-'+dt+'-'+str(size)];candidate=data['candidate-'+dt+'-'+str(size)];reference=data['pytorch-'+dt+'-'+str(size)]
            gm=next(c for c in parent['diagnostic']['curves'] if c['path']=='GM Read + Write')
            lines.append(f'| {dt}／{size}³ | {parent["facts"]["duration_us"]:.3f} | {candidate["facts"]["duration_us"]:.3f} | {reference["facts"]["duration_us"]:.3f} | {gm["reference_achievement"]:.1%} |')
    lines+=['',
        'PyTorch这里仅采GEMM组件，候选包含bias和激活；这些时间用于机制诊断，不是完整算子加速比。正式性能使用下面的同合同KGS结果。采集时间与独立设备计时也不混合相减。',
        '',
        '1024³ FP16父版GM请求约3.001MiB，PyTorch约4.000MiB，父版仍更慢；父版L2流量约30.055MiB，PyTorch约24.006MiB。父版Cube活跃比例约0.143，PyTorch约0.675。这说明继续盲目减少HBM请求缺少依据，需检查片内搬运、缓冲压力与流水调度。参考曲线的limiting_side只说明曲线较低的一侧，不能把远低于曲线的kernel直接判为计算受限。',
        '',
        '最终候选1024³ FP16的GM约3.001MiB，基本不变；L2流量升至约38.078MiB，比父版增加约26.7%，设备采集时间由36.281μs升至43.201μs。2048³同样出现L2流量增加和变慢。这与单输出分块失去一部分片内复用相符，反对采用该变换；不能仅凭这组相关性断言某一条指令是唯一原因。逐block活跃比例的样本数量随program数变化，不能直接把均值变化当成整卡利用率变化。',
        '',
        '## 修改与真实验证','',
        'Codex根据数值证据与已有源码／IR提出：将双N输出tile改为单输出tile，尝试减少两dot调度与中间缓冲。受控模板保留其他形状、FP32与非连续输入的父路径。',
        '',
        '| 阶段 | 实际结果 |','|---|---|',
        '| 首轮5配置 | 4项UB编译超限；1项因相对profiler路径缺少task CSV，不能判定性能 |',
        '| 修复计时后3配置 | 设备相对父版0.768×、0.733×、0.644×，全部更慢 |',
        '| CC＋DeepSeek追加1配置 | 选择64×128、K256，设备相对父版0.481×，未采纳 |',
        f'| 开发集最佳候选完整KGS | {comparison["candidate_geomean"]:.3f}× PyTorch，42/42 |',
        f'| 同批父版完整KGS | {comparison["parent_geomean"]:.3f}× PyTorch，42/42 |',
        f'| 同case真实比较 | 候选／父版实际加速{comparison["actual_candidate_over_parent"]:.3f}×；保留父版 |',
        '',
        '完整合同为27项正确性＋15项性能，各源码一次正式评测；补充18项覆盖三dtype、非整除小形状、非连续输入、输入不修改与stream。大连续分支另有开发集1024／2048的FP16和BF16正确性检查。报告的相对父版提升由逐case真实延迟重算，未用两组相对PyTorch汇总分数相除替代。',
        '',
        '## KG接入与模型贡献','',
        '新增build-kg-overlay.py的--roofline-evidence参数，把已校验的紧凑摘要注入两种原生KG profiling分析角色，保留角色、输入与摘要SHA。本轮在宿主专用tmux中真实调用CC＋DeepSeek消费该角色：一次选择追加候选，一次在GM口径修订后给出保留父版的裁决。模型没有直接修改源码；模板、受控空间、计数校正、实验编排和最终核验由Codex完成。没有重新执行一轮完整KG自主搜索。',
        '',
        '模型回执原样保留，不能当作已核验事实。例如修订后回执曾把曲线compute侧较低直接说成当前kernel计算受限，且推断无同case采集的BF16机制；这些说法没有被报告采纳。最终以实际流水数据、源码／IR、同输入采集与KGS结果裁决。',
        '',
        '父版包含此前KG／HG-KDA与Codex优化及共享兼容修复，不是原版FlagGems。两组使用同一KGS benchmark fingerprint；本轮不更新历史FlagGems对照成绩。',
        '',
        '## 交付文件与复现','',
        '- validation/delivered.py：本轮保留的父版源码；SHA为'+comparison['source_sha256']+'。',
        '- validation/comparison.json：同case正式延迟、选择规则与通过数。',
        '- analysis-final/：12组成对数值证据、manifest、points、教师分析器输出与中文图表。',
        '- final-roofline-summary.json：可注入新KG分析角色的摘要。',
        '- search、search-v2、search-v3：编译失败、计时缺口、成功计时与候选源码。',
        '- agent-review、agent-review-v2、kg-overlay相关身份：提示词、真实provider回执与角色快照。',
        '- empirical-audit/：4项经验参考的原task CSV与独立重算。',
        '- evidence-audit.json、transfers/：独立核验与原件包逐文件SHA。',
        '',
        '~~~bash',
        '# 在固定910B容器中，先确认项目服务空闲并持物理卡7项目锁',
        'cd /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1',
        'source /usr/local/Ascend/cann-9.0.0/set_env.sh',
        'export ASCEND_RT_VISIBLE_DEVICES=7',
        'python tools/evaluation/evaluate-operator.py --operator matmul_bias_activation \\',
        '  --source operators/campaigns/roofline-20261027/validation/delivered.py \\',
        '  --output /data/hanle/ascend-optimization/runs/新的复验目录 --label roofline-parent \\',
        '  --server http://127.0.0.1:19659 --repeats 1 --timing-scope device_kernel',
        '~~~','',
        '采集与离线分析命令见tools/profiling/hgkda/roofline/README.md；先设置项目PYTHONPATH，使固定FlagGems和KGS client可见。完整采集耗时与普通评测不同，不在其他项目任务使用卡7时并行启动。',
        '',
        '本轮所有失败与旧分析目录保留；analysis-parent-v3及其早期角色摘要曾混用逻辑复制带宽，已被同GM事件模型的analysis-parent-v4和analysis-final取代，不用于最终数值结论。无新模拟器或RTL采集。此次只使用项目卡锁、专用tmux和目录，未修改共享镜像或服务；不提交／推送GitHub。','']
    out.write_text('\n'.join(lines),encoding='utf8')
    with (b/'正式逐case对比.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['case_id','axes','parent_ms','candidate_ms','speedup']);writer.writeheader();writer.writerows(comparison['cases'])
    print(json.dumps(dict(report=str(out),selected=comparison['selected'],speedup=comparison['actual_candidate_over_parent'])))

if __name__=='__main__':main()
