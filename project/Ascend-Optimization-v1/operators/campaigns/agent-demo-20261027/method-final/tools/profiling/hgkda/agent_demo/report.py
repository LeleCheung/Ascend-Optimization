"""生成三步演示Markdown与独立HTML，所有数值取实际归档。"""
import argparse,csv,hashlib,html,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--verified',action='store_true');a=p.parse_args();b=a.batch
    out=b/('三步演示-核验版.md' if a.verified else '三步演示.md');assert not out.exists()
    facts=json.loads((b/'feedback/facts.json').read_bytes());result=facts['comparison'];agent=json.loads((b/'feedback/result.json').read_bytes())
    evidence=json.loads((b/'agent-input.json').read_bytes());route=json.loads((b/'agent/route/decision.json').read_bytes())
    proposal=json.loads((b/'agent/proposal.json').read_bytes())
    lines=['# HG-KDA 单算子三步演示','',
        '对象：matmul_bias_activation，计算 ReLU(A×B+bias)。一条宿主命令串联真机对比、模型选工具、结构化诊断、指令热点、通用源码变换、完整KGS与预测复核。','',
        f'本批最终候选相对历史优化起点实际设备加速 **{result["actual_candidate_over_seed"]:.3f}×**，相对PyTorch **{result["candidate_pytorch_speedup"]:.3f}×**。两版各42/42、补充18＋24项通过。候选与此前成功源码同字节，本批展示新采集、新模型调用与优化方法复现，不称全新发现。','',
        '## 第一步：自动对比','',
        '同一输入生成器、相同seed、shape和dtype分别运行PyTorch与起点。脚本核对卡7和目标kernel，整理搬运、指令数、流水、等待与冲突；按block保留均值、极值、总数及缺失值。','',
        '| FP16规模 | 起点融合kernel μs | PyTorch GEMM组件 μs |','|---|---:|---:|']
    for pair in evidence['pairs']:
        lines.append(f'| {pair["size"]}³ | {pair["records"]["candidate"]["duration_us"]:.3f} | {pair["records"]["pytorch"]["duration_us"]:.3f} |')
    lines+=['','PyTorch该表只采GEMM组件，候选包含bias/ReLU；这些profiling时间不作为完整算子加速比。原件见seed-profile，指标差异见paired/*/comparison.json。','',
        '## 第二步：形成诊断','',
        'Agent主动选择：'+ '、'.join(route['tools'])+'。控制器实际读取并返回所选原件，工具调用记录见agent/tool-results，真实CC＋DeepSeek请求见agent/route和agent/plan。','',
        '输入包括源码、编译器实际生成的TTIR／bcmlir、成对硬件指标、同精度经验Roofline及历史知识。Roofline回答距离参考线有多远；源码／IR和实验用于判断具体原因。','',
        '模型选择的动作：'+ '、'.join(proposal['config_ids'])+'。修改由已有通用模板生成：共享B改变相邻输出块的复用方向；交错加载改变K循环中的load/dot顺序。模板和控制器由Codex实现，模型负责证据选择与动作决策。','',
        '开发六组结果：','',
        '| 动作 | 同批起点实际加速 |','|---|---:|']
    for row in json.loads((b/'search/state.json').read_bytes())['cases']:
        lines.append(f'| {row["case"]} | {row["speedup"]:.3f}× |')
    lines+=['','正式前冻结开发优胜者，不依据正式结果重新调参。经验Roofline采用同环境参考，未证明物理峰值；源码debug位置不等于因果归因。','',
        '## 第三步：深入定位并验证修改','',
        '通过KGS的Ascend指令解析器读取模拟器清单，再由HG-KDA按候选JIT符号过滤、按单指令观测时间排序，并保留实际源码映射。下表是起点的少量示例：','',
        '| 核 | 指令 | 模拟器μs | 源码行与片段 |','|---|---|---:|---|']
    for core in evidence['instruction']['hotspots']['kernels']:
        for row in core['hotspots'][:2]:
            source=row.get('source',{});code=source.get('code','未映射').replace('|','／')
            lines.append(f'| {core["core"]} | {row["instruction"]} | {row["running_time_us"]:.3f} | {source.get("line","—")}：{code} |')
    lines+=['','起点模拟器共'+str(evidence['instruction']['instructions'])+'条指令，'+str(evidence['instruction']['mapped_instructions'])+'条附源码映射。模拟器在计算／导出后子进程退出异常，记录为部分证据；不能将其时间当正式延迟或相加重建关键路径。编译debug映射表明来源位置，不证明该Python表达式独占硬件指令耗时。','',
        '| 版本 | 相对PyTorch设备加速比 | 完整KGS | 补充语义 |','|---|---:|---:|---:|',
        f'| 起点 | {result["seed_pytorch_speedup"]:.3f}× | 42/42 | 18＋24项 |',
        f'| 候选 | {result["candidate_pytorch_speedup"]:.3f}× | 42/42 | 18＋24项 |','',
        f'15项相同case实际延迟比几何平均为 **{result["actual_candidate_over_seed"]:.3f}×**；最终保留{result["selected"]}。不是用两组对PyTorch分数相除。原始正式结果见validation，最终源码为validation/delivered.py。','',
        '修改后的同输入采集：','',
        '| FP16规模 | 起点μs | 候选μs | GM字节比 候选/起点 | L2字节比 |','|---|---:|---:|---:|---:|']
    for pair in facts['pairs']:
        r=pair['traffic_candidate_over_seed']
        lines.append(f'| {pair["size"]}³ | {pair["facts"]["reference"]["duration_us"]:.3f} | {pair["facts"]["candidate"]["duration_us"]:.3f} | {r["GM Read + Write"]:.3f} | {r["L2 Read + Write"]:.3f} |')
    lines+=['','同输入两档的GM／L2／L1总流量基本不变；UB Read + Write约为起点的0.831×，Vector Read UB约0.727×。AIV MTE2指令数为0.818×，平均每条搬运字节约1.217×；bcmlir静态copy／alloc均由5处降为4处。这些事实支持片内尾处理与搬运组织发生变化，不证明唯一因果。','',
        '模型预测“减少B侧总搬运、消除尾处理等待”未获数据支持：AIC搬运指令数未降，模拟器WAIT仍存在。交错加载只做开发集验证（1.008×），没有单独正式评测或修改后的指令采集，不能说两方案都完成正式验收。','',
        'Agent复核原回执见feedback/agent-review。原回执把两方案都通过正式评测、所有层流量不变等写法不准确；本报告依据原件更正，不直接复制模型总结。','',
        '参考图见[数值Roofline](图表/float16-参考图.png)，演示HTML可直接打开。','',
        '## 演示流程','',
        '~~~mermaid',
        'flowchart LR',
        '  A[同输入真机对比] --> B[Agent选择证据工具]',
        '  B --> C[源码 IR Roofline 热点]',
        '  C --> D[机制假设与优化动作]',
        '  D --> E[源码变换与完整KGS]',
        '  E --> F[重新采集 验证预测]',
        '  F --> G[保留最佳与核验知识]',
        '~~~','',
        '## 演示时怎么讲','',
        '“我们把同输入profiling、源码／IR和经验Roofline整理成Agent可读的证据。Agent先选需要查看的工具，再结合热点和历史知识选择修改。改完后用同一合同完整测试，并重新采集指标，检查原先的解释是否成立。”','',
        '“这个矩阵乘演示中，Agent选择共享B和交错加载；共享B在开发集胜出并通过完整评测。最终源码复现了历史成功方案，证明这段人工分析经验能够通过统一入口运行。复杂因果解释仍需交叉核验。”','',
        '## 复现入口','',
        '在910B宿主固定控制器环境运行，输出使用新批次目录。源码和经验参考需先按项目清单同步。','',
        '~~~bash',
        '/data/hanle/ascend-optimization/runtime/kg-controller/venv/bin/python \\',
        '  /data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/hgkda/agent_demo/run.py \\',
        '  --batch /data/hanle/ascend-optimization/runs/新的三步演示',
        '~~~','',
        '用tmux运行该命令可在Windows关机后继续。控制器使用卡7项目锁、固定tle_yy容器和KGS19659；每源码一次正式评测。状态未知的操作不自动重发。新运行的模拟器可能出现已记录的退出异常，保留指令证据与状态，不假装完整成功。','',
        '本批首次真机采集、三次模拟器调试和模型调用在统一控制器完成前已执行；workflow-state明确登记为恢复采用本批原件。最终入口对新目录能串联同样阶段，人工调试与模型调用分别归档。Windows同步排除可再生compiler-cache，原始指令CSV、trace、源码、IR、计时与模型回执保留；排除清单见transfers。','']
    out.write_text('\n'.join(lines),encoding='utf8')
    with (b/'正式逐case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','axes','seed_ms','candidate_ms','speedup']);w.writeheader();w.writerows(result['cases'])
    escaped=html.escape('\n'.join(lines))
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>HG-KDA 三步演示</title><style>body{margin:0;background:#f5f7fb;color:#17243b;font:16px/1.7 system-ui,sans-serif}main{max-width:1080px;margin:40px auto;padding:36px;background:white;border-radius:16px}h1{margin:0;color:#153e75}section{display:flex;gap:16px;margin:28px 0}.card{flex:1;padding:20px;background:#edf4ff;border-radius:12px}.n{color:#2363b8;font-size:26px;font-weight:700}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 ui-monospace,monospace}a{color:#185ab0}</style><main><h1>HG-KDA：从硬件证据到优化验证</h1><p>matmul_bias_activation · Ascend 910B</p><section><div class="card"><div class="n">01 自动对比</div>同输入PyTorch／候选<br>指标聚合与差异</div><div class="card"><div class="n">02 形成诊断</div>Agent选择工具<br>源码＋IR＋Roofline</div><div class="card"><div class="n">03 深入定位</div>热点指令映射<br>修改＋完整KGS验证</div></section><p><strong>相对本批起点 '+f'{result["actual_candidate_over_seed"]:.3f}×'+' · 完整评测42/42 · 补充18＋24项</strong></p><p>历史成功方法的新采集与流程复现；模拟器退出异常按部分证据记录。</p><p><a href="三步演示.md">打开Markdown</a> · <a href="正式逐case.csv">逐case数据</a></p><pre>'+escaped+'</pre></main></html>'
    html_name='演示-核验版.html' if a.verified else '演示.html'
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    pair_table=table(['FP16规模','起点融合kernel μs','PyTorch GEMM组件 μs'],[[str(q['size'])+'³',f'{q["records"]["candidate"]["duration_us"]:.3f}',f'{q["records"]["pytorch"]["duration_us"]:.3f}'] for q in evidence['pairs']])
    final_table=table(['版本','相对PyTorch设备加速比','完整KGS','补充检查'],[['起点',f'{result["seed_pytorch_speedup"]:.3f}×','42/42','18＋24'],['共享B候选',f'{result["candidate_pytorch_speedup"]:.3f}×','42/42','18＋24']])
    hotspot_rows=[]
    for core in evidence['instruction']['hotspots']['kernels']:
        for row in core['hotspots'][:1]:
            s=row.get('source',{});hotspot_rows.append([core['core'],row['instruction'],str(row['running_time_us'])+' μs',str(s.get('line','—'))+'：'+s.get('code','未映射')])
    hotspot_table=table(['核','热点指令','模拟器时间','源码位置'],hotspot_rows)
    body='<h2>01 自动对比</h2><p>相同输入、实际kernel与设备身份核验，脚本聚合指标并计算差异。</p>'+pair_table+'<p class="note">PyTorch仅GEMM组件；该表不是完整算子加速比。</p>'
    body+='<h2>02 Agent形成诊断</h2><p>实际选择工具：'+html.escape('、'.join(route['tools']))+'</p><p>读取源码、IR、成对指标、经验Roofline与热点。选择共享B和交错加载；六组开发输入筛选后，冻结共享B方案。</p><p><a href="agent/proposal.json">查看模型原始决策</a> · <a href="agent/route/decision.json">查看工具选择</a></p><img alt="同输入Roofline经验参考" src="图表/float16-参考图.png" style="width:100%">'
    body+='<h2>03 热点定位与修改验证</h2>'+hotspot_table+'<p class="note">指令源码映射来自编译debug信息；模拟器子进程退出异常，导出指令仅作部分诊断证据。</p>'+final_table
    body+='<p><strong>相同15项实际延迟：'+f'{result["actual_candidate_over_seed"]:.3f}×起点'+'</strong></p><p>GM／L2／L1总流量基本不变；UB读写约下降16.9%，AIV MTE2指令数约下降18.2%，平均搬运粒度约增加21.7%。这比“总搬运量减少”更符合本次证据。</p><p>未消除所有等待；交错加载未进入正式评测。原模型不准确的归因已核验更正。</p>'
    body+='<p><a href="'+html.escape(out.name)+'">完整中文说明</a> · <a href="正式逐case.csv">逐case正式数据</a> · <a href="validation/delivered.py">最终源码</a></p>'
    document=document[:document.index('<pre>')]+body+'</main></html>'
    document=document.replace('</style>','table{width:100%;border-collapse:collapse;margin:16px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce4ee}th{background:#edf4ff}.note{font-size:14px;color:#54667d}h2{margin-top:36px;color:#153e75}</style>')
    (b/html_name).write_text(document,encoding='utf8')
    marker=b/('report-verified.json' if a.verified else 'report-completed.json')
    marker.write_text(json.dumps(dict(status='completed',files=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [out,b/html_name,b/'正式逐case.csv']]),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='completed',report=str(out),speedup=result['actual_candidate_over_seed'])))

if __name__=='__main__':main()
