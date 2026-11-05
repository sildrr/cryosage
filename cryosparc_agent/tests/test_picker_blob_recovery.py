import json
from types import SimpleNamespace

import pytest

from picker_blob_recovery import DiameterAwareRecovery, validate_diagnosis
from test_sequential_picker_rollout import thresholds, gate, serial_instance


def row():
    return dict(count=100,box_size_pix=576,picker_parameters=dict(diameter=120,diameter_max=300),
        thresholds=thresholds(),picker='old_picker',inspect='old_inspect',extract='old_extract')


def test_blob_range_must_change_with_valid_physical_bounds():
    for p in [dict(diameter=120,diameter_max=300),dict(diameter=300,diameter_max=120),
              dict(diameter=0,diameter_max=200),dict(diameter=100),dict(diameter=True,diameter_max=220)]:
        with pytest.raises(ValueError):
            validate_diagnosis(dict(target='blob',parameters=p,reason='test'),row(),[])
    good=dict(target='blob',parameters=dict(diameter=100,diameter_max=220),reason='new calibrated evidence')
    assert validate_diagnosis(good,row(),[]) is good
    with pytest.raises(ValueError):
        validate_diagnosis(good,row(),[dict(picker_parameters=good['parameters'])])


def test_old_interventions_keep_their_contracts():
    good=dict(target='inspect',parameters=thresholds(.3),reason='membership evidence')
    assert validate_diagnosis(good,row(),[])['target']=='inspect'
    with pytest.raises(ValueError):
        validate_diagnosis(dict(target='extract',parameters=dict(box_size_pix=576),reason='unchanged'),row(),[])


def test_extract_explanation_is_normalized_without_mutating_input():
    scale = dict(diameter_max_A=220, confidence='uncertain', basis='calibrated evidence')
    source = dict(target='extract', parameters=dict(box_size_pix=440,
        scale_estimate=scale, outside_preference_reason='test'), reason='change box')
    normalized = validate_diagnosis(source, row(), [])
    assert normalized['parameters'] == dict(box_size_pix=440)
    assert normalized['scale_estimate'] == scale
    assert normalized['outside_preference_reason'] == 'test'
    assert 'scale_estimate' in source['parameters']


def test_extract_conflicting_evidence_and_unsafe_parameters_are_rejected():
    with pytest.raises(ValueError, match='Conflicting'):
        validate_diagnosis(dict(target='extract', parameters=dict(box_size_pix=448,
            scale_estimate={'diameter_max_A': 220}), scale_estimate={'diameter_max_A': 300},
            reason='change box'), row(), [])
    for parameters in (dict(box_size_pix=448, bin_size_pix=2),
                       dict(box_size_pix=721), dict(box_size_pix=441)):
        with pytest.raises(ValueError):
            validate_diagnosis(dict(target='extract', parameters=parameters,
                                    reason='change box'), row(), [])


def instance(tmp_path):
    e=serial_instance(tmp_path)
    e.__class__=DiameterAwareRecovery
    return e


def test_blob_revision_redecides_inspect_and_extract_in_order(tmp_path,monkeypatch):
    import vision_inputs
    e=instance(tmp_path)
    events=[]
    e.sample_picker=lambda count:events.append('new_blob') or SimpleNamespace(uid='new_picker')
    e.request_inspect=lambda *args:events.append('new_thresholds') or thresholds(.4)
    e.inspect_serial=lambda *args:(events.append('inspect_review') or SimpleNamespace(uid='new_inspect'),thresholds(.4))
    monkeypatch.setattr(vision_inputs,'build_extract_box_visual_context',lambda *args,**kwargs:{})
    e.stage=lambda *args:events.append('new_box') or dict(box_size_pix=440)
    e.extract_serial=lambda *args:(events.append('extract_review') or SimpleNamespace(uid='new_extract'),440)
    e.classify=lambda tag,ex,box,picker,inspected,count,params:dict(
        box_size_pix=box,thresholds=params,picker_parameters=e.state['picker_parameters'])
    decision=dict(target='blob',parameters=dict(diameter=100,diameter_max=220),reason='wrong original scale')
    result=e.apply_adjustment('new_branch',row(),decision)
    assert events==['new_blob','new_thresholds','inspect_review','new_box','extract_review']
    assert result['box_size_pix']==440
    assert result['thresholds']==thresholds(.4)
    assert e.state['picker_parameters']==decision['parameters']


def test_completed_recovery_row_resumes_without_resubmitting(tmp_path):
    e=instance(tmp_path)
    e.scales=(50,100,200)
    old=row()
    ready=dict(old,executable=True,classes='Jnew',evidence_slug='new',gate=gate())
    e.state.update(sequential_trials=dict(old=dict(row=old)),diameter_recovery_trials={
        'diameter_recovery_r0_n100':dict(row=ready,reconstruction=dict(gate=gate(),
            bank_job='bank',best_refine='refine',selections=[]))})
    p=tmp_path/'evidence/new/context.json';p.parent.mkdir(parents=True);p.write_text('{}')
    e.request_adjustment=lambda *args:pytest.fail('replayed model diagnosis')
    e.sample_picker=lambda *args:pytest.fail('replayed picker')
    assert e.pilot()['best_refine']=='refine'


def test_recovered_full_run_does_not_restart_pilot(tmp_path):
    e=instance(tmp_path)
    e.state.update(rollout_complete=True,best_refine='existing_refine')
    e.pilot=lambda:pytest.fail('seed replay')
    assert e.run()=='existing_refine'
