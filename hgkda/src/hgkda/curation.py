"""Evidence-bound knowledge curation: deduplication, scope filtering, validation gates, and bounded retrieval."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from .artifacts import unwrap
from .validation import supplement_status
from .gate import admit

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def organize(records, *, root=Path('.')):
    groups = {}
    rejected = []
    for record in records:
        row = dict(record)
        required = ['operator_type', 'action_id', 'scope', 'summary', 'evidence']
        if any((not row.get(key) for key in required)):
            rejected.append(dict(record=row, reason='Missing structured scope or evidence'))
            continue
        if row.get('status') in {'validated', 'observed_failure'}:
            try:
                expect_failure = row['status'] == 'observed_failure'
                failures = 0
                ev = row['evidence']
                source = root / ev['source']
                formal = root / ev['formal_result']
                result = json.loads(formal.read_text(encoding='utf8'))
                if not digest(source) == ev['source_sha256']:
                    raise ValueError('Evidence identity or validation result does not satisfy admission requirements')
                if not digest(formal) == ev['formal_result_sha256']:
                    raise ValueError('Evidence identity or validation result does not satisfy admission requirements')
                if not (result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads'] and (result['num_workloads'] > 0)):
                    raise ValueError('Evidence identity or validation result does not satisfy admission requirements')
                request = root / ev['formal_request']
                payload = json.loads(request.read_text(encoding='utf8'))
                sources = payload['implementation']['sources']
                if not (len(sources) == 1 and sources[0]['content'] == source.read_text(encoding='utf8')):
                    raise ValueError('Evidence identity or validation result does not satisfy admission requirements')
                if row.get('record_type', 'candidate') == 'candidate':
                    review = root / ev['backend_review']
                    if not digest(review) == ev['backend_review_sha256']:
                        raise ValueError('Evidence identity or validation result does not satisfy admission requirements')
                    admit(sources[0]['content'], json.loads(review.read_text(encoding='utf8')))
                for extra in ev.get('supplementary', []):
                    p = root / extra['path']
                    if not digest(p) == extra['sha256']:
                        raise ValueError('Evidence identity or validation result does not satisfy admission requirements')
                    d = json.loads(p.read_text(encoding='utf8'))
                    if not d['source_sha256'] == ev['source_sha256']:
                        raise ValueError('Evidence identity or validation result does not satisfy admission requirements')
                    if expect_failure:
                        if not d['status'] in {'passed', 'failed'}:
                            raise ValueError('Evidence identity or validation result does not satisfy admission requirements')
                        failures += d['status'] == 'failed'
                    else:
                        if not d['status'] == 'passed':
                            raise ValueError('Evidence identity or validation result does not satisfy admission requirements')
                        operator = row['scope'].get('operator', d.get('operator'))
                        if not supplement_status(operator, d, ev['source_sha256']) == 'passed':
                            raise ValueError('Supplementary checks do not satisfy the current operator gate')
                if expect_failure:
                    if ev.get('performance_feedback'):
                        item=ev['performance_feedback'];path=root/item['path']
                        if digest(path)!=item['sha256']:raise ValueError('Performance feedback identity changed')
                        document=json.loads(path.read_bytes());feedback=unwrap(document)
                        bound={value.get('sha256') for value in document.get('inputs',[])}
                        threshold=row['scope'].get('performance_threshold')
                        if not (isinstance(threshold,(int,float)) and math.isfinite(threshold) and threshold>0
                            and ev['source_sha256'] in bound and ev['formal_result_sha256'] in bound
                            and feedback.get('geo_mean')==result.get('geo_mean')
                            and feedback.get('threshold')==threshold
                            and feedback.get('decision')=='continue_optimization'
                            and feedback.get('accepted') is False
                            and 0<result['geo_mean']<=threshold):
                            raise ValueError('Performance failures require matching source, complete scores, and an explicit scoped threshold')
                        failures+=1
                    if not failures > 0:
                        raise ValueError('Failure constraint lacks verifiable failed supplementary checks or performance results')
            except (KeyError, AssertionError, OSError, ValueError) as error:
                row['status'] = 'draft'
                row['validation_error'] = type(error).__name__ + ': ' + str(error)
        elif row.get('status') not in {'draft', 'rejected', 'stale'}:
            row['status'] = 'draft'
        key = json.dumps([row['operator_type'], row['action_id'], row['scope']], sort_keys=True, ensure_ascii=False)
        groups.setdefault(key, []).append(row)
    active = []
    history = []
    for key, rows in groups.items():
        rows.sort(key=lambda r: (r.get('status') in {'validated', 'observed_failure'}, r.get('updated_at', '')), reverse=True)
        chosen = rows[0]
        chosen['dedup_key'] = hashlib.sha256(key.encode()).hexdigest()
        active.append(chosen)
        for row in rows[1:]:
            row['superseded_by'] = chosen.get('id', chosen['dedup_key'])
            history.append(row)
    return dict(schema='hgkda-knowledge-v1', active=active, superseded=history, rejected=rejected, evidence_root=str(root.resolve()), policy='Deduplicate by type/action/scope; mark validated or observed_failure only after source/request/result verification. Failure constraints require matching failed supplementary checks or complete performance feedback with scoped thresholds. Preserve prior originals.')

def retrieve(index, operator_type, scope, max_chars=6000):
    selected = []
    used = 0
    checked = organize(index['active'], root=Path(index.get('evidence_root', '.')))
    for row in checked['active']:
        if row['status'] not in {'validated', 'observed_failure'} or row['operator_type'] != operator_type:
            continue
        if any((scope.get(k) != v for k, v in row['scope'].items() if v != '*')):
            continue
        text = json.dumps(row, ensure_ascii=False)
        if used + len(text) > max_chars:
            continue
        selected.append(row)
        used += len(text)
    return dict(records=selected, characters=used, budget=max_chars, policy='Retrieve only evidence-verified implementations or failure constraints with compatible scope; never truncate individual evidence records')

def implementation_examples(index, kind, scope, *, max_chars=3000):
    """Recommend inspectable source across operators without transferring scores or generalizing implementation facts."""
    from .validation import supplement_status
    checked = organize(index.get('active', []), root=Path(index.get('evidence_root', '.')))
    selected = []
    used = 0
    for row in sorted(checked['active'], key=lambda r: r.get('updated_at', ''), reverse=True):
        if row.get('status') != 'validated' or row['operator_type'] != kind:
            continue
        if row['scope'].get('operator') == scope.get('operator'):
            continue
        if any((scope.get(k) != v for k, v in row['scope'].items() if k != 'operator' and v != '*')):
            continue
        ev = row['evidence']
        extras = [json.loads(Path(x['path']).read_text(encoding='utf8')) for x in ev.get('supplementary', [])]
        if not extras or any((supplement_status(row['scope'].get('operator'), e, ev['source_sha256']) != 'passed' for e in extras)):
            continue
        entry = dict(operator=row['scope'].get('operator'), source=ev['source'], source_sha256=ev['source_sha256'], qualification='The source implementation passed its own full contract and supplementary checks; a new operator still needs independent evaluation', usage='Inspect launch, tiling, and layout handling as needed; formulas and legal tile sizes cannot be copied blindly, and performance does not transfer', evidence=ev)
        size = len(json.dumps(entry, ensure_ascii=False))
        if used + size > max_chars:
            continue
        selected.append(entry)
        used += size
        if len(selected) == 2:
            break
    return dict(examples=selected, characters=used, budget=max_chars)
