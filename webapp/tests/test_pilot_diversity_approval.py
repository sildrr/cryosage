import copy
import json
from pathlib import Path
import tempfile
import unittest
from webapp.pilot_approval import approved_diversity, candidates, diversity_only, fingerprint, request_approval


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.gate = dict(available=True, passed=False, qualified_particles=2500,
            supported_group_count=2, failed_conditions=['insufficient_supported_2d_shape_groups'])

    def test_particle_floor_and_other_errors_never_waived(self):
        self.assertTrue(diversity_only(self.gate))
        for patch in [dict(qualified_particles=2000), dict(qualified_particles=float('nan')),
                      dict(available=False), dict(failed_conditions=[]),
                      dict(failed_conditions=['insufficient_supported_2d_shape_groups', 'other_error'])]:
            self.assertFalse(diversity_only(dict(self.gate, **patch)))

    def test_requires_explicit_approval_and_keeps_failed_measurement(self):
        self.assertFalse(approved_diversity(self.gate))
        gate = dict(self.gate, manual_diversity_approval=dict(scope='pilot_shape_diversity_only',
            approved=True, id='id', approved_at='date', reason='human request', fingerprint='sha'))
        self.assertTrue(approved_diversity(gate))
        self.assertFalse(gate['passed'])
        self.assertFalse(approved_diversity(dict(gate, qualified_particles=1999)))

    def test_candidates_rank_completed_reconstruction_and_reject_low_quality(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'manifest.json').write_text(json.dumps(dict(production_pipeline='web-pilot-to-full-v1')))
            def record(res, particles):
                return dict(row=dict(executable=True, classes='J1', count=200), reconstruction=dict(
                    gate=dict(self.gate, qualified_particles=particles), selections=[{'job':'J2'}],
                    best_refine='J3', best_resolution_A=res))
            state = dict(status='protocol_exhausted', sequential_trials={
                'best':record(3.5, 2500), 'worse':record(4.0, 5000), 'bad_quality':record(3.0, 1000)})
            (root/'state.json').write_text(json.dumps(state))
            self.assertEqual([r['tag'] for r in candidates(root)], ['best','worse'])
            approval=request_approval(root,'User explicitly approved continuation','test')
            self.assertEqual(approval['tag'],'best')
            self.assertEqual(approval['fingerprint'],fingerprint(state['sequential_trials']['best']))
            self.assertTrue((root/'pilot_diversity_approvals'/f"{approval['id']}.json").exists())
            state['pilot_accepted']={'classes':'J1'}
            (root/'state.json').write_text(json.dumps(state))
            self.assertEqual(candidates(root),[])

    def test_fingerprint_detects_parameter_change(self):
        record={'row':{'box_size_pix':528}}
        changed=copy.deepcopy(record);changed['row']['box_size_pix']=504
        self.assertNotEqual(fingerprint(record),fingerprint(changed))

    def test_low_quality_has_no_candidates_or_approval(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'manifest.json').write_text(json.dumps(dict(production_pipeline='web-pilot-to-full-v1')))
            (root/'state.json').write_text(json.dumps(dict(status='protocol_exhausted',
                sequential_trials={'bad':dict(row=dict(executable=True,classes='J1',count=200),
                reconstruction=dict(gate=dict(self.gate,qualified_particles=2000),
                    selections=[{'job':'J2'}],best_refine='J3',best_resolution_A=3.0))})))
            self.assertEqual(candidates(root),[])
            with self.assertRaises(ValueError):
                request_approval(root,'Human would like to continue')
            self.assertFalse((root/'pilot_diversity_approval.json').exists())


if __name__ == '__main__':
    unittest.main()
