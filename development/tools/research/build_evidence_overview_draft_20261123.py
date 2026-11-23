"""Preview the shared Analyst/Coder role names in the evidence overview."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from build_simulator_flow_draft_20261122 import DELIVERY, MAIN, text

ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'development/reports/evidence-overview-draft-20261123'
MATERIAL=DELIVERY/'material/01_开场与总览'
STEM='HG-KDA-overview-角色命名草案'


def node(x,y,w,h,lines,dark=False):
    values=[f'<g class="node"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="17" fill="{"#133F6D" if dark else "#E7EFF7"}" stroke="{"#133F6D" if dark else "#CAD5E2"}" stroke-width="1.6"/>']
    for value,offset in lines:
        values.append(text(x+w/2,y+offset,value,26,color='#FFFFFF' if dark else '#133F6D',weight=600))
    return '\n'.join(values+['</g>'])


def main():
    before=hashlib.sha256(MAIN.read_bytes()).hexdigest()
    REPORT.mkdir(parents=True,exist_ok=True)
    MATERIAL.mkdir(parents=True,exist_ok=True)
    svg='\n'.join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="title desc">',
        '<title id="title">HG-KDA：Hardware-Grounded Evidence</title>',
        '<desc id="desc">当前算子从成对profiling、模拟器分析和知识库/代码推理取得依据，由Analyst Agent分析、Coder Agent修改，编译验证后输出优化版本，并继续反馈迭代。</desc>',
        '<defs><marker id="arrow" markerWidth="13" markerHeight="13" refX="11" refY="6.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L12 6.5 L0 13 Z" fill="#606870"/></marker>',
        '<marker id="feedback" markerWidth="13" markerHeight="13" refX="11" refY="6.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L12 6.5 L0 13 Z" fill="#133F6D"/></marker></defs>',
        '<rect width="1600" height="900" fill="#FFFFFF"/>',
        '<g font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">',
        text(65,90,'HG-KDA：Hardware-Grounded Evidence',44,color='#14181C',weight=700,anchor='start'),
        '<g fill="none" stroke="#606870" stroke-width="2.8" stroke-linejoin="round">',
        '<path d="M250 450 H285"/>',
        '<path d="M285 265 V635"/>',
        *[f'<path d="M285 {y} H330" marker-end="url(#arrow)"/>' for y in (265,450,635)],
        *[f'<path d="M670 {y} H720"/>' for y in (265,450,635)],
        '<path d="M720 265 V635"/>',
        '<path d="M720 450 H775" marker-end="url(#arrow)"/>',
        '<path d="M1025 450 H1080" marker-end="url(#arrow)"/>',
        '<path d="M1280 450 H1335" marker-end="url(#arrow)"/>',
        '</g>',
        '<path d="M1180 500 V790 H155 V500" fill="none" stroke="#133F6D" stroke-width="2.4" stroke-dasharray="10 10" marker-end="url(#feedback)"/>',
        text(790,770,'反馈迭代',26,weight=500),
        node(60,400,190,100,[('当前算子',60)],True),
        node(330,215,340,100,[('成对 profiling',60)]),
        node(330,400,340,100,[('模拟器分析',60)]),
        node(330,585,340,100,[('知识库 / 代码推理',60)]),
        node(775,375,250,150,[('Analyst Agent',57),('Coder Agent',105)],True),
        node(1080,400,200,100,[('编译与验证',60)]),
        node(1335,400,210,100,[('优化后算子',60)],True),
        '</g></svg>',
    ])
    mermaid='''flowchart LR
    IN[当前算子] --> P[成对profiling]
    IN --> S[模拟器分析]
    IN --> K[知识库 / 代码推理]
    P --> A["Analyst Agent<br/>Coder Agent"]
    S --> A
    K --> A
    A --> V[编译与验证] --> OUT[优化后算子]
    V -. 反馈迭代 .-> IN
'''
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HG-KDA / 角色命名草案</title><style>*{box-sizing:border-box}body{margin:0;background:#edf1f5;color:#14181c;font:17px/1.65 "Microsoft YaHei",sans-serif}main{max-width:1640px;margin:auto;padding:18px 20px 40px}nav{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:16px}a{color:#133f6d;text-decoration:none;background:white;border:1px solid #cad5e2;border-radius:7px;padding:6px 16px}.slide{background:#fff;border-radius:10px;overflow:hidden}.slide svg{display:block;width:100%;height:auto}.note{background:white;border:1px solid #d7e0e9;border-radius:9px;padding:18px 24px;margin-top:18px}@media print{nav,.note{display:none}main{padding:0}}</style></head><body><main><nav><a href="../06_成对Profiling/成对profiling-简图草案.html">技术点1 / 成对profiling</a><a href="../03_模拟器/模拟器分析-简图草案.html">技术点2、3 / 模拟器</a></nav><section class="slide">__SVG__</section><p class="note">统一使用Analyst Agent与Coder Agent表达分析与实现职责。总图保留输入、三路依据、修改、验证、输出与反馈关系；本草案未修改主PPT，也不额外声称独立多Agent编排。</p></main></body></html>'''.replace('__SVG__',svg)
    for suffix,content in (('.svg',svg),('.html',page),('.mmd',mermaid)):
        path=REPORT/(STEM+suffix)
        path.write_text(content,encoding='utf-8')
        shutil.copy2(path,MATERIAL/path.name)
    assert hashlib.sha256(MAIN.read_bytes()).hexdigest()==before
    (REPORT/'validation.json').write_text(json.dumps({'status':'generated','main_ppt_preserved':True,'main_ppt_sha256':before,'delivery':str(MATERIAL/(STEM+'.html'))},ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(MATERIAL/(STEM+'.html')))


if __name__=='__main__':main()
