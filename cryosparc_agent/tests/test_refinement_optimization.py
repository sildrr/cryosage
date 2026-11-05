import json
from types import SimpleNamespace

import numpy as np
import pytest

import refinement_optimization as opt
from benchmark_budget import protect_client, BenchmarkBudgetReached
from experiment_planner import build_policy_plan, apply_anti_refinement_loop


class Population:
    def __init__(self, **fields):
        self.data = {key: np.asarray(value) for key, value in fields.items()}
    def __getitem__(self, name):
        return self.data[name]
    def __len__(self):
        return len(self.data['uid'])
    def take(self, indices):
        return Population(**{key: value[indices] for key, value in self.data.items()})
    def fields(self):
        return list(self.data)


def node(uid, kind, parents=(), title='', res=None):
    return dict(cryosparc_job_uid=uid, workflow_node_id=uid, job_type=kind,
        status='completed', title=title, parent_job_uids=list(parents),
        key_parameters={'refine_symmetry': 'C1'}, outputs={
            'particles': {'available': True}, 'volume': {'available': True,
                'latest_summary_stat_values': {'fsc_info': {'radwn_final_A': res}}}})


def state():
    return {'nodes': [node('J1', 'class_2D_new'), node('J2', 'select_2D', ['J1']),
                      node('J3', 'homo_refine_new', ['J2'], res=2.6)]}


def test_offers_optics_scale_and_reference_routes(monkeypatch):
    monkeypatch.setattr(opt, 'reference_sampling', lambda *args: 128)
    s = state()
    actions = opt.build_candidates(s, s['nodes'][-1], 'P1')
    assert {a['rollback_kind'] for a in actions} == {
        'optics_refinement', 'low_scale_cleanup', 'reference_class_expansion'}
    optics = actions[0]
    assert optics['default_parameters']['crg_do_spherical'] is True
    assert optics['default_parameters']['refine_defocus_refine'] is True
    assert optics['parameter_template']['refine_symmetry']['const'] == 'C1'
    assert optics['refinement_strategy_switch'] is True


def test_imported_particles_need_no_picking_or_extract_ancestors(monkeypatch):
    monkeypatch.setattr(opt, 'reference_sampling', lambda *args: 320)
    s = state()
    s['nodes'].insert(0, node('J0', 'import_particles'))
    s['nodes'][1]['parent_job_uids'] = ['J0']
    actions = opt.build_candidates(s, s['nodes'][-1], 'P1')
    reference = next(a for a in actions if a['rollback_kind'] == 'reference_class_expansion')
    assert reference['required_inputs'] == {'volume': [dict(source_job_uid='J3', source_output='volume')]}
    assert reference['default_parameters']['vol_resample_N'] == 320
    assert any(a['rollback_kind'] == 'optics_refinement' for a in actions)


def test_no_recursive_scale_cut_and_no_repeated_optics(monkeypatch):
    monkeypatch.setattr(opt, 'reference_sampling', lambda *args: 128)
    s = state()
    s['nodes'][1]['title'] = 'Agent opt_scale_J0'
    s['nodes'][-1]['key_parameters'].update(refine_defocus_refine=True, refine_ctf_global_refine=True)
    actions = opt.build_candidates(s, s['nodes'][-1], 'P1')
    assert [a['rollback_kind'] for a in actions] == ['reference_class_expansion']


def test_reference_continuation_preserves_explicit_sources():
    s = state()
    low = node('J4', 'volume_tools', ['J3'], 'Agent opt_reference_lowpass_J3_J1_J2')
    s['nodes'].append(low)
    match = opt.continuation(s, low)[0]
    assert match['job_type'] == 'reference_select_2D'
    assert match['required_inputs']['volume'][0]['source_job_uid'] == 'J4'
    n = node('J5', 'reference_select_2D', ['J4', 'J1'], 'Agent opt_reference_match_J3_J1_J2')
    s['nodes'].append(n)
    union = opt.continuation(s, n)[0]
    assert union['job_type'] == 'agent_reference_union'
    assert union['required_inputs']['original'][0]['source_job_uid'] == 'J2'
    n = node('J6', 'agent_reference_union', ['J1', 'J2', 'J3', 'J5'], 'Agent opt_reference_union_J3_J1_J2')
    s['nodes'].append(n)
    refine = opt.continuation(s, n)[0]
    assert refine['job_type'] == 'nonuniform_refine_new'
    assert refine['required_inputs']['particles'][0]['source_output'] == 'selected'


def test_union_never_drops_original_and_restores_source_metadata():
    source = Population(uid=[1, 2, 3, 4], tag=[11, 22, 33, 44])
    selected, audit = opt.select_population('agent_reference_union', dict(source=source,
        original=Population(uid=[1, 3]), proposed=Population(uid=[2, 3])), {})
    assert selected['uid'].tolist() == [1, 2, 3]
    assert selected['tag'].tolist() == [11, 22, 33]
    assert audit['added_count'] == 1
    with pytest.raises(ValueError):
        opt.select_population('agent_reference_union', dict(source=source,
            original=Population(uid=[1]), proposed=Population(uid=[9])), {})


def test_scale_cut_uses_current_p05_preserves_splits_and_rejects_ties():
    source = Population(**{'uid': np.arange(2000), 'alignments3D/alpha_min': np.linspace(.1, 2, 2000),
                           'alignments3D/split': np.arange(2000) % 2})
    selected, audit = opt.select_population('agent_scale_cleanup', {'particles': source}, {'quantile': .05})
    assert len(selected) == 1900
    assert np.array_equal(selected['alignments3D/split'], selected['uid'] % 2)
    assert audit['threshold'] == pytest.approx(np.quantile(source['alignments3D/alpha_min'], .05))
    source.data['alignments3D/alpha_min'][:] = 1.
    with pytest.raises(ValueError):
        opt.select_population('agent_scale_cleanup', {'particles': source}, {'quantile': .05})


def test_optics_survives_upstream_guard_and_is_preferred_in_fallback(monkeypatch):
    monkeypatch.setattr(opt, 'reference_sampling', lambda *args: 128)
    s = state()
    actions = opt.build_candidates(s, s['nodes'][-1], 'P1')
    observation = {'observed_issues': [{'issue_type': 'target_resolution_unmet'}]}
    plan = build_policy_plan(observation, actions, [], 1)
    assert plan['approved_action_ids'] == ['opt_optics_J3']
    guarded = apply_anti_refinement_loop(plan, observation, actions, [], 1)
    assert guarded['approved_action_ids'] == ['opt_optics_J3']


def test_external_job_creation_counts_against_same_budget(tmp_path, monkeypatch):
    budget = tmp_path / 'budget.json'
    budget.write_text(json.dumps(dict(project='P1', workspace='W1', maximum=1)))
    monkeypatch.setenv('CRYOAGENT_BENCHMARK_BUDGET_FILE', str(budget))
    class Client:
        def __init__(self):
            self.jobs = []
        def create_job(self, project_uid, workspace_uid, type):
            self.jobs.append(SimpleNamespace(uid='J' + str(len(self.jobs) + 1)))
            return self.jobs[-1]
        def create_external_job(self, project_uid, workspace_uid, title='', desc=''):
            self.jobs.append(SimpleNamespace(uid='J' + str(len(self.jobs) + 1)))
            return self.jobs[-1]
        def find_project(self, project_uid):
            return SimpleNamespace(find_workspace=lambda uid: SimpleNamespace(find_jobs=lambda: self.jobs))
    client = protect_client(Client())
    client.create_external_job('P1', 'W1')
    with pytest.raises(BenchmarkBudgetReached):
        client.create_job('P1', 'W1', type='nonuniform_refine_new')
    assert len(client.jobs) == 1
    assert len(list((tmp_path / 'budget_receipts').glob('*.json'))) == 1


def test_external_inputs_always_have_explicit_slots():
    source = Population(**{'uid': [1, 2, 3], 'blob/idx': [0, 1, 2]})
    datasets = {'J1': source, 'J2': source.take([0]), 'J3': source.take([1])}
    members = [SimpleNamespace(uid=uid, title=uid, status='completed',
        load_output=lambda output, uid=uid: datasets[uid]) for uid in ['J1', 'J2', 'J3']]
    members.append(SimpleNamespace(uid='J4', title='J4', status='completed'))
    connected = []
    class External:
        uid = 'J5'
        def connect(self, name, uid, output, *, slots):
            assert slots
            connected.append((name, slots))
        def add_output(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def save_output(self, name, value):
            self.saved = value
        def load_output(self, name):
            return self.saved
        def log(self, text):
            pass
        def stop(self, error=''):
            assert not error
    workspace = SimpleNamespace(find_jobs=lambda: members)
    project = SimpleNamespace(find_workspace=lambda uid: workspace)
    cs = SimpleNamespace(find_project=lambda uid: project, create_external_job=lambda *args, **kwargs: External())
    plan = dict(job_type='agent_reference_union', action_id='opt_reference_union_J4_J1_J2',
        resolved_parameters={}, connections=dict(source=('J1', 'particles'),
            original=('J2', 'selected'), proposed=('J3', 'selected'), volume=('J4', 'volume')))
    result = opt.execute_particle_operation(cs, 'P1', 'W1', plan)
    assert result['success'] is True
    assert result['audit']['selected_count'] == 2
    assert ('volume', ['map']) in connected
