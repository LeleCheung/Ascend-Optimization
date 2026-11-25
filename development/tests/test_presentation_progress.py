"""Check that chart selection preserves measured meaning and source relations."""
import copy
import re
import unittest
import xml.etree.ElementTree as ET

from development.tools.traces.progress_view import improving
from development.tools.traces.research_progress import milestones, render


class PresentationProgressTest(unittest.TestCase):
    def test_failed_native_and_target_regression_preserve_raw_records(self):
        data = dict(operator='example', metric='fixed_case_cann_speedup', events=[
            dict(label='FlagGems', qualified=False, score=None),
            dict(label='V1', qualified=True, score=.5, geo_mean=1.1),
            dict(label='V2', qualified=True, score=.4, geo_mean=1.8),
            dict(label='V3', qualified=True, score=.8, geo_mean=1.9)])
        original = copy.deepcopy(data)
        self.assertEqual(milestones(data)['displayed_indices'], [0, 1, 3])
        self.assertEqual(data, original)
        svg = render(data)
        self.assertIn('Fixed shape 0.800', svg)
        self.assertIn('source-lineage', svg)
        self.assertNotIn('0.400×', svg)

    def test_curve_segments_connect_after_a_hidden_regression(self):
        data = dict(operator='example', events=[dict(label=str(i), score=value, qualified=True)
            for i, value in enumerate([.5, 1.0, .8, 1.2])])
        root = ET.fromstring(render(data))
        segments = [[float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', node.attrib['d'])]
            for node in root.iter() if node.attrib.get('class') == 'retained-best']
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0][2:], segments[1][:2])
        self.assertTrue(all(s[3] <= s[1] for s in segments))

    def test_summary_keeps_method_transfer_distinct_from_source_ancestry(self):
        events = [dict(id=str(i), score=score) for i, score in enumerate([.3, .2, .8])]
        row = dict(display_events=events, connections=[
            dict(start='0', end='1', kind='recorded_lineage'),
            dict(start='1', end='2', kind='documented_method_switch')])
        selected = improving(row)
        connection = selected['connections'][0]
        self.assertEqual(connection['kind'], 'documented_method_switch')
        self.assertEqual(len(connection['proof']), 2)
        self.assertEqual(row['display_events'], events)

    def test_higher_remeasurement_is_hidden_and_ir_edit_keeps_its_round(self):
        data = dict(operator='example', events=[
            dict(label='FlagGems', qualified=True, score=.3, source_sha256='native'),
            dict(label='source', qualified=True, score=.5, source_sha256='best'),
            dict(label='original IR replay', qualified=True, score=.6, source_sha256='best',
                 ir_sha256='original-ir', measurement_transition=True,
                 measurement_transition_label='same source replay'),
            dict(label='IR improvement', qualified=True, score=.55, source_sha256='best',
                 ir_sha256='changed-ir')])
        original = copy.deepcopy(data)
        selection = milestones(data)
        self.assertEqual(selection['displayed_indices'], [0, 1, 3])
        self.assertEqual(selection['detail_indices'], [2])
        self.assertEqual(selection['axis_labels']['3'], 'Round 2')
        drawing = render(data)
        self.assertNotIn('same source replay', drawing)
        self.assertNotIn('measurement-transition', drawing)
        self.assertNotIn('original-ir', drawing)
        self.assertIn('changed-ir', drawing)
        self.assertEqual(data, original)

    def test_lower_remeasurement_stays_in_details_without_plot_annotation(self):
        data = dict(operator='example', events=[
            dict(label='FlagGems', qualified=True, score=.3, measurement_event_id='native'),
            dict(label='source best', qualified=True, score=.5, measurement_event_id='source'),
            dict(label='original IR replay', qualified=True, score=.48, measurement_event_id='replay',
                 measurement_transition=True, same_protocol_remeasurement=True),
            dict(label='IR improvement', qualified=True, score=.55, measurement_event_id='ir',
                 measurement_bridge_event_id='replay')])
        original = copy.deepcopy(data)
        self.assertEqual(milestones(data)['displayed_indices'], [0, 1, 3])
        self.assertEqual(milestones(data)['detail_indices'], [2])
        self.assertNotIn('Same-source repeat', render(data))
        self.assertEqual(data, original)

    def test_unmarked_source_remeasurement_retains_original_connection_proof(self):
        row = dict(display_events=[
            dict(id='native', score=.3, source_sha256='native'),
            dict(id='source', score=.5, source_sha256='best'),
            dict(id='replay', score=.6, source_sha256='best'),
            dict(id='ir', score=.55, source_sha256='best', ir_sha256='changed-ir')],
            connections=[dict(start=a, end=b, kind=kind) for a, b, kind in
                [('native', 'source', 'recorded_lineage'), ('source', 'replay', 'same_source'),
                 ('replay', 'ir', 'recorded_lineage')]])
        original = copy.deepcopy(row)
        view = improving(row)
        self.assertEqual([e['id'] for e in view['display_events']], ['native', 'source', 'ir'])
        self.assertEqual(view['presentation_hidden_ids'], ['replay'])
        self.assertEqual(len(view['connections'][-1]['proof']), 2)
        self.assertEqual(row, original)


if __name__ == '__main__':
    unittest.main()
