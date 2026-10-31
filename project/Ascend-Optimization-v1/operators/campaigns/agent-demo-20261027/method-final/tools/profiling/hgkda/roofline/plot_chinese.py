"""绘制中文经验参考图；标明组件范围与参考不确定性。"""
import argparse,json,math
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('不覆盖已有图表')
    a.output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names={f.name for f in font_manager.fontManager.ttflist}
    choices=[n for n in ['Microsoft YaHei','SimHei','Noto Sans CJK SC','WenQuanYi Zen Hei'] if n in names]
    if choices:plt.rcParams['font.sans-serif']=choices
    plt.rcParams['axes.unicode_minus']=False
    data=json.loads(a.analysis.read_bytes());labels={'parent':'父版融合kernel','candidate':'候选融合kernel','pytorch':'PyTorch GEMM组件'}
    colors={'parent':'#225ea8','candidate':'#d95f0e','pytorch':'#238b45'}
    for dt in ['float16','float32']:
        cases=[r for r in data['cases'] if r['spec']['dtype']==dt]
        if not cases:continue
        fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
        for ax,path,title in zip(axes,['GM Read + Write','L2 Read + Write'],['GM经验参考','L2模型参考']):
            points=[(r,next(c for c in r['diagnostic']['curves'] if c['path']==path)) for r in cases]
            maxx=max(c['algorithmic_ai'] for _,c in points)*2
            xs=[10**(i/80*math.log10(maxx)) for i in range(81)]
            first=points[0][1];roof=[min(first['reference_compute_tflops'],x*first['reference_bandwidth_gbps']/1000) for x in xs]
            ax.loglog(xs,roof,'--',color='#666666',label='同精度经验计算＋对应带宽参考')
            seen=set()
            for r,c in points:
                side=r['spec']['side'];label=labels[side] if side not in seen else None;seen.add(side)
                ax.scatter(c['algorithmic_ai'],c['algorithmic_tflops'],s=65,color=colors[side],label=label)
                ax.annotate(str(r['spec']['shape'][0]),(c['algorithmic_ai'],c['algorithmic_tflops']),xytext=(5,5),textcoords='offset points',fontsize=8)
            ax.set(xlabel='有效2MNK / 官方该层字节（FLOP/Byte）',ylabel='有效算法吞吐（TFLOP/s）',title=title)
            ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=8,loc='lower right')
        fig.suptitle('Ascend 910B：'+dt+' 矩阵乘同输入对照',fontsize=14)
        fig.supxlabel('经验参考不是已证明物理上界；PyTorch仅GEMM组件，完整算子性能以KGS为准。',fontsize=10)
        fig.savefig(a.output/(dt+'-参考图.png'),dpi=170)
        fig.savefig(a.output/(dt+'-参考图.svg'))
        plt.close(fig)
    (a.output/'图表说明.md').write_text('# 图表说明\n\n横轴是有效算法2MNK除以官方该层流量，纵轴是有效算法吞吐。虚线分别组合同精度经验GEMM吞吐与同PMU模型persistent copy参考／官方L2带宽模型。它们用于同输入诊断，不代表已标定的物理峰值。PyTorch仅采GEMM组件，候选包含bias和激活，不能由图直接给出完整算子加速比。\n',encoding='utf8')

if __name__=='__main__':main()
