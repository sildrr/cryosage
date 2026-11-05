import json
from types import SimpleNamespace

import numpy as np
import pytest

from pilot_objective_gate import GatePolicy
from two_tier_picker import two_tier_gate, TwoTierPicker, PROTOCOL, gate_policy


def entry(scores=(1.5, 2.0, 3.5), counts=(1500, 1501, 400), duplicate=False):
    rng = np.random.default_rng(123)
    images = {i:rng.normal(size=(64,64)) for i in range(len(scores))}
    if duplicate:
        images[2] = images[0].copy()
    context = dict(source=dict(project_uid='P6',job_uid='J1'),
        class_statistics=[dict(class_id=i,particle_count=n) for i,n in enumerate(counts)],
        cryosift=dict(success=True,project_uid='P6',job_uid='J1',
            class_scores=[dict(class_id=i,score=s) for i,s in enumerate(scores)]))
    return context,images,list(range(len(scores)))


def test_new_threshold_is_scoped_and_historical_policy_is_preserved():
    assert gate_policy({}).duplicate_ncc == .90
    assert gate_policy({'gate': {'duplicate_ncc': .85}}).duplicate_ncc == .85
    policy = gate_policy({'gate': {'duplicate_ncc': .90}})
    assert policy.min_particles == 2001
    assert policy.min_groups == 3 and policy.min_group_fraction == .05
    assert GatePolicy().duplicate_ncc == .85


def test_higher_threshold_separates_similar_but_not_identical_shapes():
    from pilot_objective_gate import shape_groups
    matrix = np.array([[1., .88, .2], [.88, 1., .2], [.2, .2, 1.]])
    assert len(shape_groups(matrix, .85)) == 2
    assert len(shape_groups(matrix, gate_policy({}).duplicate_ncc)) == 3
    assert len(shape_groups(np.ones((3, 3)), .90)) == 1


def test_explicit_gate_reassessment_reuses_branch_without_mutating_old_gate():
    old = dict(available=True, qualified_particles=3000, supported_group_count=2)
    new = dict(available=True, qualified_particles=3000, supported_group_count=3)
    amendment = dict(bucket='sequential_trials', tag='pilot_r2_n200', duplicate_ncc=.90)
    original = dict(classes='J3014', gate=old, evidence_slug='old')
    state = dict(sequential_trials=dict(pilot_r2_n200=dict(row=original)))
    def reconstruct(tag, row):
        assert tag == amendment['tag'] and row['gate'] is new
        assert row['classes'] == 'J3014'
        return dict(gate=new, bank_job='Jbank', best_refine='J3033', selections=[])
    e = SimpleNamespace(manifest=dict(pilot_gate_reassessment=amendment), state=state,
        policy=gate_policy({}), project=SimpleNamespace(find_job=lambda uid:uid),
        readout=lambda slug,job:({},new), core_and_recovery=reconstruct, persist=lambda:None)
    accepted = TwoTierPicker.pilot(e)
    assert accepted['best_refine'] == 'J3033'
    assert original['gate'] is old and original['evidence_slug'] == 'old'
    assert state['gate_reassessment']['amendment'] == amendment


def test_gate_reassessment_rejects_failed_gate_or_full_data_replacement():
    e = SimpleNamespace(manifest=dict(pilot_gate_reassessment=dict(
        bucket='sequential_trials', tag='trial', duplicate_ncc=.90)),
        policy=gate_policy({}), state=dict(full=dict(classes='J1')))
    with pytest.raises(ValueError, match='active full-data'):
        TwoTierPicker.pilot(e)
    e.state = dict(sequential_trials=dict(trial=dict(row=dict(classes='J2'))))
    e.project = SimpleNamespace(find_job=lambda uid:uid)
    e.readout = lambda *a:({},dict(available=True,qualified_particles=3000,supported_group_count=2))
    with pytest.raises(ValueError, match='does not pass'):
        TwoTierPicker.pilot(e)


def test_supplement_can_supply_third_group_but_not_core_particles():
    g = two_tier_gate([entry()], {'P6/J1':[2]}, GatePolicy(min_particles=2001))
    assert g['passed'] and g['supported_group_count'] == 3
    assert g['qualified_particles'] == 3001
    assert g['retained_particles'] == 3401 and g['supplemental_particles'] == 400
    assert g['metric_version'] == PROTOCOL


def test_legacy_combined_gate_remains_strict():
    from sequential_picker_rollout import combine_gate
    old = combine_gate([entry()], GatePolicy(min_particles=2001))
    assert not old['passed'] and old['supported_group_count'] == 2
    assert old['qualified_particles'] == 3001


def test_weak_particles_cannot_replace_the_quality_floor():
    g = two_tier_gate([entry(counts=(1000,1000,400))], {'P6/J1':[2]}, GatePolicy(min_particles=2001))
    assert not g['passed'] and g['qualified_particles'] == 2000
    assert 'insufficient_strict_core_particles' in g['failed_conditions']


def test_score_four_is_not_accepted_and_reference_scope_is_required():
    with pytest.raises(ValueError):
        two_tier_gate([entry(scores=(1.,2.,4.))], {'P6/J1':[2]}, GatePolicy())
    with pytest.raises(ValueError, match='reference'):
        two_tier_gate([entry()], {'P6/OTHER':[2]}, GatePolicy())


def test_duplicate_shape_and_low_support_do_not_count_as_third_view():
    g = two_tier_gate([entry(duplicate=True)], {'P6/J1':[2]}, GatePolicy())
    assert not g['passed'] and g['supported_group_count'] == 2
    g = two_tier_gate([entry(counts=(1500,1501,10))], {'P6/J1':[2]}, GatePolicy())
    assert not g['passed'] and g['supported_group_count'] == 2


def test_source_classes_cannot_be_counted_twice():
    with pytest.raises(ValueError, match='Duplicate source'):
        two_tier_gate([entry(),entry()], {'P6/J1':[2]}, GatePolicy())


def test_score_exactly_three_is_supplement_not_core():
    g = two_tier_gate([entry(scores=(1.,2.,3.))], {'P6/J1':[2]}, GatePolicy())
    assert g['qualified_particles'] == 3001 and g['supplemental_particles'] == 400


def test_template_diameter_uses_this_datasets_pixel_size():
    e = TwoTierPicker.__new__(TwoTierPicker)
    e.state = dict(pilot_accepted=dict(box_size_pix=400,selections=[]))
    e.native_pixel_size_A = 1.058
    e.persist = lambda: None
    def model(slug,messages,model,validate):
        assert '423.2000 A' in messages[1]['content'][0]['text']
        assert '1.058 A/px' in messages[1]['content'][0]['text']
        return validate(dict(diameter=200,reason='calibrated'))
    e.model = model
    assert e.template_parameters()['diameter'] == 200


@pytest.mark.parametrize('pixel,dose',[(1.058,80.),(.82,30.255),(.856,68.4)])
def test_stage_uses_dataset_specific_acquisition_facts(tmp_path,monkeypatch,pixel,dose):
    from pilot_reclassification_experiment import PilotExperiment
    import kb_bridge
    import visual_decision_contract
    captured = {}
    def messages(stage,context,**kwargs):
        captured.update(context)
        return []
    monkeypatch.setattr(kb_bridge,'call_kb_tool',lambda *a,**kw:{})
    monkeypatch.setattr(visual_decision_contract,'build_stage_decision_messages',messages)
    e = SimpleNamespace(root=tmp_path,acquisition_facts=dict(psize_A=pixel,accel_kv=300.,cs_mm=2.7,total_dose_e_per_A2=dose),
        service=lambda label,fn:fn(),write=lambda *a:None,
        model=lambda slug,msg,model,validate:validate(dict(parameters=dict(diameter=100,diameter_max=200))))
    PilotExperiment.stage(e,'test','blob_picker_diameter',{})
    assert captured['acquisition_facts']['psize_A'] == pixel
    assert captured['acquisition_facts']['total_dose_e_per_A2'] == dose
