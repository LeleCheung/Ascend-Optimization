"""Audit the displayed recovery deliverable without treating scheme switches as ancestry."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import xml.etree.ElementTree as ET


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.urls=[]
    def handle_starttag(self,tag,attrs):
        self.urls.extend(v for k,v in attrs if k in ('href','src'))


def main():
    root=Path.cwd();out=root/'development/reports/operator-traces-stitched-20261117'
    rows=json.loads((out/'cann-view.json').read_bytes())['operators'];assert len(rows)==26
    summaries=[]
    for row in rows:
        events=row['display_events'];connections=row['connections']
        assert len(connections)==len(events)-1
        for a,b,c in zip(events,events[1:],connections):
            assert (a['id'],b['id'])==(c['start'],c['end'])
            assert c['kind'] in ('same_source','recorded_lineage','recorded_recovery','documented_method_switch')
            if c['kind']!='same_source':assert c['proof']
            if c['kind']=='documented_method_switch':assert c['source_inheritance'] is False
        assert all(e['score'] is not None or e['is_baseline'] for e in events)
        earlier=[e['score'] for e in events[:-1] if e['score'] is not None]
        assert not earlier or events[-1]['score']>=max(earlier)
        folder=out/row['operator'];svg=ET.fromstring((folder/'mainline.svg').read_bytes())
        nodes=[e for e in svg.iter() if e.attrib.get('class')=='node'];assert len(nodes)==len(events)
        links=[e for e in svg.iter() if e.attrib.get('class') in ('connection','method-switch')]
        assert len(links)==len(connections)
        assert all('stroke-dasharray' not in e.attrib for e in links)
        assert (folder/'mainline.png').stat().st_size>0
        if row.get('recovery'):
            p=root/row['recovery']['evaluation'];audit=json.loads((p.parent.parent/'audit-final.json').read_bytes())['rows'][0]
            assert audit['full_passed'] and audit['source_sha256']==row['recovery']['source_sha256']
            assert all(c['status']=='passed' and c['passed']==c['total'] for c in audit['checks'].values())
            source=p.parent.parent/'candidate.py';assert hashlib.sha256(source.read_bytes()).hexdigest()==audit['source_sha256']
        summaries.append(dict(operator=row['operator'],nodes=len(events),
            direct_native_lineage=row['full_lineage_verified'],method_transfer=bool(row.get('method_transfer')),
            final_score=events[-1]['score'],recovery=bool(row.get('recovery'))))
    for p in out.rglob('*.html'):
        if p.name.startswith('raw-'):continue
        links=Links();links.feed(p.read_text(encoding='utf8'))
        for url in links.urls:
            if ':' in url or url.startswith('#'):continue
            assert (p.parent/url.split('#')[0]).exists(),(p,url)
    result=dict(status='passed',scope='26 displayed progress curves, explicit ancestry versus donor switches, final nonregression, new-source complete evaluations, local links and exports',
        limitations=['Fixed-reference historical rebasing does not remove cross-time device noise.',
                     'Four operators use donor implementations; not 26 direct same-operator ancestry chains.',
                     'Previously issued delivery ZIP and operators selections are unchanged.',
                     'Valid intermediate negative trials remain in history.'],operators=summaries)
    (out/'completion-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status='passed',operators=26,direct_lineage=sum(x['direct_native_lineage'] for x in summaries),
                         donor_routes=sum(x['method_transfer'] for x in summaries),recoveries=sum(x['recovery'] for x in summaries))))


if __name__=='__main__':main()
