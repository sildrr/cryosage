import asyncio
import sys
from types import SimpleNamespace

import pytest

from upstream_evidence import collect_current_pilot, lineage_facts, review_picker, template_picker_visual_context


def install_project(monkeypatch, *, missing_extract=False, foreign=False):
    records = {
        'J1': dict(uid='J1', parents=[], spec=dict(type='blob_picker_gpu', params={'diameter': 100})),
        'J2': dict(uid='J2', parents=['J1'], spec=dict(type='inspect_picks_v2', params={'ncc_score_thresh': .1})),
        'J3': dict(uid='J3', parents=['J2'], spec=dict(type='extract_micrographs_multi', params={'box_size_pix': 384})),
        'J4': dict(uid='J4', parents=['J2' if missing_extract else 'J3'], spec=dict(type='class_2D_new', params={})),
        'J5': dict(uid='J5', parents=['J4'], spec=dict(type='select_2D', params={'selected_templates': '1,3'})),
    }
    for uid, data in records.items():
        data['workspace_uids'] = ['W9' if foreign and uid == 'J1' else 'W1']
    project = SimpleNamespace(find_job=lambda uid: SimpleNamespace(
        model=SimpleNamespace(model_dump=lambda: records[uid])))
    monkeypatch.setitem(sys.modules, 'cryosparc_client', SimpleNamespace(
        cryosparc_client=lambda: SimpleNamespace(find_project=lambda _: project)))


@pytest.mark.parametrize('current,stage,extract,classes', [
    ('J2', 'inspect_picks_v2', None, None),
    ('J3', 'extract_micrographs_multi', 'J3', None),
    ('J4', 'class_2D_new', 'J3', 'J4'),
    ('J5', 'select_2D', 'J3', 'J4'),
])
def test_stage_aware_lineage_never_requires_future_jobs(monkeypatch, current, stage, extract, classes):
    install_project(monkeypatch)
    result = lineage_facts('P1', current, 'W1', 'J1')
    assert result['review_stage'] == stage
    assert result['inspect_job_uid'] == 'J2'
    assert result['extract_job_uid'] == extract
    assert result['class2d_job_uid'] == classes
    if extract is None:
        assert result['extract_parameters'] is None


def test_class_review_still_requires_completed_lineage(monkeypatch):
    install_project(monkeypatch, missing_extract=True)
    with pytest.raises(ValueError, match='Unexpected Inspect/Extract'):
        lineage_facts('P1', 'J4', 'W1', 'J1')


def test_foreign_workspace_is_still_rejected(monkeypatch):
    install_project(monkeypatch, foreign=True)
    with pytest.raises(ValueError, match='leaves the assigned workspace'):
        lineage_facts('P1', 'J2', 'W1', 'J1')


def test_select_templates_review_is_recognized():
    assert review_picker({'candidate_actions':[{'diameter_trial':{
        'role':'select_templates', 'picker_uid':'J1'}}]}) == 'J1'


def test_review_uses_current_branch_not_first_historical_promotion():
    candidates = {'candidate_actions': [
        {'diameter_trial': dict(role='accept_full', picker_uid='Jold', class_job_uid='Joldclass')},
        {'diameter_trial': dict(role='accept_full', picker_uid='Jnew', class_job_uid='Jnewclass')},
    ]}
    assert review_picker(candidates, 'Jnewclass') == 'Jnew'
    candidates['candidate_actions'].append({'pilot_template_recovery': dict(pilot_picker='Jnew')})
    assert review_picker(candidates, 'Jselect') == 'Jnew'
    candidates['candidate_actions'][-1] = {'diameter_trial': dict(role='select_templates', picker_uid='Jnew')}
    assert review_picker(candidates, 'Jnewclass') == 'Jnew'


def test_template_picker_receives_actual_class_images_and_selected_labels():
    calls = []
    async def call(session, tool, arguments):
        calls.append((tool, arguments))
        return {'contact_sheet': {'data_url': 'data:image/png;base64,test'}} if tool == 'get_class_average_visual_context' else {'success': True}
    context = asyncio.run(template_picker_visual_context(call, None, 'P1', {
        'candidate_actions': [{'pilot_template_recovery': dict(source_class_job='J4',
             source_select_job='J5', selected_templates='1,3', pilot_picker='J1')}]
    }, cryosift=True))
    assert context['kind'] == 'pilot_template_picker'
    assert context['selected_templates'] == '1,3'
    assert context['contact_sheet']['data_url'].startswith('data:image/')
    assert calls[0][1] == dict(project_uid='P1', job_uid='J4', max_classes=0)
    assert calls[1][0] == 'evaluate_2d_classes_with_cryosift'


def test_select_lineage_reports_the_actual_selected_templates(monkeypatch):
    install_project(monkeypatch)
    assert lineage_facts('P1', 'J5', 'W1', 'J1')['selected_templates'] == '1,3'


def test_2d_templates_do_not_attempt_to_load_a_nonexistent_volume(monkeypatch):
    import volume_scale
    job = SimpleNamespace(type='select_2D')
    project = SimpleNamespace(find_job=lambda uid: job)
    monkeypatch.setattr(volume_scale, 'cryosparc_client', lambda: SimpleNamespace(find_project=lambda _: project))
    result = volume_scale.estimate_template_picker_volume_scale('P1', 'J5')
    assert result['applicable'] is False
    assert 'failure_reason' not in result


def test_unavailable_optional_lineage_is_reported_not_fabricated():
    async def call(*args):
        raise AssertionError('No evidence calls allowed for invalid lineage')
    def facts(*args):
        raise ValueError('Invalid lineage')
    result = asyncio.run(collect_current_pilot(call, None, 'P1', 'W1', 'J2',
        {'candidate_actions':[{'diameter_trial':{'role':'retry','picker_uid':'J1'}}]}, facts_loader=facts))
    assert result['success'] is False
    assert result['attachments'] == []
    assert result['error']['message'] == 'Invalid lineage'


def test_transport_errors_are_not_silently_swallowed():
    def facts(*args):
        raise ConnectionError('Service unavailable')
    with pytest.raises(ConnectionError):
        asyncio.run(collect_current_pilot(None, None, 'P1', 'W1', 'J2',
            {'candidate_actions':[{'diameter_trial':{'role':'retry','picker_uid':'J1'}}]}, facts_loader=facts))
