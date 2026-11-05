import json

import pytest

from extract_scale_context import scale_memory, validate_scale_response, neighbor_summary
from sequential_picker_rollout import SequentialPickerRollout


def original():
    return dict(parameters=dict(box_size_pix=440),reason='Full-field estimate 180-210 A; 440px spans 376.64 A.',
                scale_estimate=dict(diameter_max_A=210,confidence='uncertain',basis='whole fields'))


def estimate(diameter=210):
    return dict(scale_estimate=dict(diameter_max_A=diameter,confidence='uncertain',basis='calibrated full-field extent'))


def test_native_scale_and_search_bounds_are_explicit():
    m=scale_memory(.856,192,dict(diameter=120,diameter_max=240),original())
    assert m['current_box_width_A']==pytest.approx(164.352)
    assert m['current_box_over_initial_estimated_diameter']==pytest.approx(164.352/210)
    assert m['picker_search_bounds']['minimum_native_pix']==pytest.approx(120/.856)
    assert m['preferred_box_range_native_pix']==pytest.approx([315/.856,420/.856])
    assert m['margin_A_each_side']<0
    assert m['original_sizing_decision']['reason']==original()['reason']


def test_soft_preference_is_explained_but_not_a_size_hard_gate():
    m=scale_memory(.856,192,original=original())
    with pytest.raises(ValueError,match='Explain a box'):
        validate_scale_response(estimate(),m,192)
    with pytest.raises(ValueError,match='Explain a box'):
        validate_scale_response(dict(estimate(),outside_preference_reason=None),m,192)
    r=validate_scale_response(dict(estimate(),outside_preference_reason='explicit exploratory diagnostic with risk'),m,192)
    assert r['scale_evaluation']['outside_soft_preference']
    assert r['scale_evaluation']['box_size_pix']==192
    assert validate_scale_response(estimate(),m,440)['scale_evaluation']['box_over_estimated_diameter']==pytest.approx(376.64/210)


def test_reduced_molecular_extent_needs_new_evidence_not_silent_bright_core_substitution():
    m=scale_memory(.856,192,original=original())
    with pytest.raises(ValueError,match='earlier molecular diameter'):
        validate_scale_response(estimate(100),m,192)
    r=validate_scale_response(dict(estimate(100),diameter_revision_reason='new calibrated whole-field evidence'),m,192)
    assert not r['scale_evaluation']['outside_soft_preference']


def test_legacy_natural_language_estimate_is_preserved_not_guessed():
    old=original();old.pop('scale_estimate')
    m=scale_memory(.856,192,dict(diameter=120,diameter_max=240),old)
    assert m['initial_estimated_max_diameter_A'] is None
    assert '180-210 A' in m['original_sizing_decision']['reason']
    with pytest.raises(ValueError,match='earlier molecular diameter'):
        validate_scale_response(estimate(100),m,192)


def test_missing_measurement_not_replaced_by_picker_lower_bound():
    m=scale_memory(.856,192,dict(diameter=120,diameter_max=240))
    r=validate_scale_response(estimate(None),m,192)
    assert r['scale_evaluation']['estimated_max_diameter_A'] is None
    assert r['scale_evaluation']['box_over_estimated_diameter'] is None


def test_neighbor_counts_are_descriptive_not_a_zero_neighbor_rule():
    samples=[dict(extra_picked_centers_by_box_px={'440':n}) for n in [0,1,2,0,4]]
    summary=neighbor_summary(samples,440)
    assert summary['boxes_with_any_extra_pick']==3
    assert summary['boxes_with_more_than_two_extra_picks']==1
    assert 'Occasional neighbors are acceptable' in summary['meaning']


def test_actual_review_receives_original_scale_reason_and_calibrated_ratio(tmp_path):
    e=SequentialPickerRollout.__new__(SequentialPickerRollout)
    e.root=tmp_path
    e.state=dict(picker_parameters=dict(diameter=120,diameter_max=240))
    e.write=lambda *a:None
    path=tmp_path/'decisions/pilot_r0_n50_box_parameters/decision.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(original()))
    captured=[]
    def model(slug,messages,name,validate):
        captured.extend(messages)
        return validate(dict(action='revise',reason='preserve full particle extent',
            parameters=dict(box_size_pix=440),**estimate()))
    e.model=model
    visual=[dict(type='text',text=json.dumps(dict(native_pixel_size_A=.856))),
            dict(type='image_url',image_url=dict(url='data:image/png;base64,AA'))]
    result=e.review_parameters('new_scale_audit','extract',visual,dict(box_size_pix=192),[])
    text=captured[1]['content'][0]['text']
    assert '180-210 A' in text
    assert 'current_box_over_initial_estimated_diameter' in text
    assert 'A few nearby automatic picks' in text
    assert 'ANGSTROMS' in text
    assert result['scale_evaluation']['box_width_A']==pytest.approx(376.64)
