"""Presentation-only filtering; measured events remain unchanged in the evidence."""


def remeasurement_indices(events):
    """Hide repeated implementations while preserving distinct IR edits."""
    hidden, seen = set(), set()
    for index, event in enumerate(events):
        source = event.get('source_sha256') or event.get('source_key')
        identity = (source, event.get('ir_sha256')) if source else None
        explicit = (event.get('measurement_transition')
            or event.get('same_protocol_remeasurement')
            or event.get('selected_layer') == 'remeasurement'
            or event.get('attribution_status') == 'version_selection_or_retest')
        if index and (explicit or (identity is not None and identity in seen)):
            hidden.add(index)
        if identity is not None:
            seen.add(identity)
    return hidden


def contract_connections(events, connections, selected):
    """Preserve every original relation crossed by a hidden measurement."""
    positions = {event['id']: i for i, event in enumerate(events)}
    original = {(c['start'], c['end']): c for c in connections}
    result = []
    for a, b in zip(selected, selected[1:]):
        span = events[positions[a['id']]:positions[b['id']] + 1]
        proof = [original.get((x['id'], y['id']), dict(start=x['id'], end=y['id'], kind='unresolved'))
            for x, y in zip(span, span[1:])]
        if len(proof) == 1:
            result.append(proof[0])
            continue
        kinds = {c['kind'] for c in proof}
        verified = kinds <= {'same_source', 'recorded_lineage', 'recorded_recovery', 'progress_summary'}
        switched = kinds <= {'same_source', 'recorded_lineage', 'recorded_recovery',
            'progress_summary', 'documented_method_switch'}
        result.append(dict(start=a['id'], end=b['id'],
            kind='progress_summary' if verified else ('documented_method_switch' if switched else 'unresolved'),
            reference_change=a.get('reference') != b.get('reference'), proof=proof,
            relation='source_ancestry' if verified else ('documented_method_transfer' if switched else 'unresolved')))
    return result
