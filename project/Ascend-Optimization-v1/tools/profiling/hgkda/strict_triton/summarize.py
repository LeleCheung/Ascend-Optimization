"""仅从最终原件生成中文复核简报；不将未完成或混合后端历史成绩写成交付。"""
from collections import Counter
from common import BATCH,read,save,sha


def main():
    delivery=read(BATCH/'delivery.json');audit=read(BATCH/'delivery-audit.json')
    assert audit['status']=='verified' and audit['delivery_sha256']==sha(BATCH/'delivery.json')
    assert delivery['queue_status']=='completed'
    assert read(BATCH/'compiled-verification.json')['status']=='completed'
    rows=delivery['rows'];accepted=[r for r in rows if r['accepted']]
    assert all(r['runtime']['status']=='passed' for r in rows)
    assert all(r['compiled_path'].get('status')=='verified' for r in rows),'实际编译路径仍有失败，先检查'
    counts=Counter(r['status'] for r in rows)
    g=delivery['groups']
    lines=['# 严格Triton 30项重新核验','',
        '**本轮严格Triton达标 %d/30：1015原10项为%d/10，1029新增20项为%d/20。**'%(len(accepted),g['1015']['accepted'],g['1029']['accepted']),'',
        '达标含义：全部设备计算、搬运和回退均使用Triton；原PyTorch正确性与补测通过；相对合格CANN参考的完整调用加速比超过0.8×。超过1×才表示快于该参考。CANN单接口、布局／精度兼容与组合参考在总表中分开标注。','',
        '固定FlagGems `d6a8eec473517a3d68157b208eb9c057eb1d4c50`、FlagTree `0.6.0+ascend.gitc286cba6`、Triton 3.5.1、CANN 9.0.0、KGS 6.5.0；KGS计时为包含host与device的完整调用。原输入、容差、参数和逐case集合未修改。','',
        '## 结果','',
        '|类别|项数|算子|','|---|---:|---|']
    for status,n in counts.items():
        operators='、'.join('`'+r['operator']+'`' for r in rows if r['status']==status)
        lines.append('|'+status+'|'+str(n)+'|'+operators+'|')
    lines+=['','完整数值、原合同通过项数及参考类型见[10＋20总表](严格Triton30项总表.md)。参考资格不足与候选数值失败分开记录，二者不能用更换参考或放宽容差消除。','',
        '## 本轮做了什么','',
        '- 从可追溯历史源码恢复Triton候选；移除12项外部设备计算路径，统一处理12项的存储入口，其中9项涉及潜在PyTorch复制、连续化、填充等，另3项只规范分配入口。实际存储工作改为内嵌Triton核；旧源码与请求均保留。',
        '- QKV恢复经逐函数审查的C++宿主胶水：只分配、取元数据与stream、打包参数和启动Triton，设备算法仍为原Triton。Python、C++和二进制身份分别绑定。',
        '- 30项完成逐路径源码／固定依赖审查、候选范围内ATen门禁、原KGS和补测以及实际Triton编译对象核验。首遍new_empty分配误报已修正，旧失败结果单列保留。',
        '- 25项合格CANN参考的原源码、请求、完整结果、补测和冻结环境重新审查通过。同字节、同合同已有CANN候选成绩明确标为复用，其余每份源码默认一次完整评测，不重测挑分。',
        '- 全部新KG角色前置严格Triton约束，知识输出分开实现模板与CANN参考。原混合后端25/30等历史统计不再作为当前目标数量。','',
        '本轮为Codex恢复、修复与验收；未新增CC／DeepSeek搜索。历史KG／HG-KDA／CC与Codex贡献沿源码来源保留，不将后端更换或测量波动称为新优化收益。','',
        '## 证据与复现','',
        '首遍selu_、rad2deg_、log10_选到了shape[0]的空正确性输入，未产生设备任务；原空trace保留，已按原合同非空性能输入补采并核验实际kernel。其余数值通过对象的设备trace正常。编译路径证据来自代表输入实际调用返回的Triton编译对象，保存TTIR、设备npubin与SHA，并结合全部分支源码和运行时审查；代表输入取证不等同于全分支硬件trace。','',
        '`fractional_max_pool3d`与`_conj_physical`的原PyTorch参考含CPU回退，该列高分不能解释为相对纯NPU库的收益。缺少合格同语义CANN参考的对象单列。','',
        '- `delivery.json`：全部候选、原始成绩、后端、参考类型、复用标记和证据SHA。',
        '- `delivery-audit.json`：30项原合同逐文件及请求bundle/case集合核验。',
        '- `reference-qualification-audit.json`：25项合格CANN参考的资格证据重核。',
        '- `strict-implementation-context.json`：已核验Triton实现与CANN参考分开输出。',
        '- `candidates/<算子>/`：冻结源码、运行时检查、完整评测、补测、实际编译产物；`revisions/`保留旧状态。',
        '- 运行入口见[严格Triton复核工具](../../../tools/profiling/hgkda/strict_triton/README.md)，使用独立目录执行一次新KGS复测。','']
    (BATCH/'复核结果简报.md').write_bytes(('\n'.join(lines)+'\n').encode('utf8'))
    save(BATCH/'result-summary.json',dict(accepted=len(accepted),total=30,groups=g,status_counts=dict(counts),
        backend_audit_passed=30,compiled_proofs=30,delivery_sha256=sha(BATCH/'delivery.json'),audit_sha256=sha(BATCH/'delivery-audit.json')))
    print('FINAL_SUMMARY',len(accepted),g,dict(counts))


if __name__=='__main__':main()
