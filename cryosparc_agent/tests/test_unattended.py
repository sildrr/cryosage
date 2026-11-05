import asyncio
import json
from types import SimpleNamespace

import pytest

import unattended
from quality_policy import build_retry_candidate
from upstream_evidence import collect_current_pilot, message_for


def failed(uid='J2', source='J1'):
    return dict(cryosparc_job_uid=uid, workflow_node_id=uid, status='failed',
                job_type='patch_ctf_estimation_multi', key_parameters={},
                inputs={'exposures': [{'source_job_uid': source, 'source_output': 'micrographs'}]})


def test_default_approval_unchanged(monkeypatch):
    monkeypatch.delenv('CRYOAGENT_UNATTENDED', raising=False)
    assert build_retry_candidate(failed())['approval_required']


def test_explicit_policy_approves_only_bounded_failed_retry(monkeypatch):
    monkeypatch.setenv('CRYOAGENT_UNATTENDED', '1')
    result = build_retry_candidate(failed(), {'nodes': [failed()]})
    assert result['approval_required'] is False
    assert result['automatic_retry']['approved']
    assert result['available']
    failures = [failed('J' + str(i)) for i in range(2, 6)]
    result = build_retry_candidate(failed(), {'nodes': failures})
    assert result['available'] is False
    assert result['blocked_by'] == ['automatic_failed_retry_limit']


def test_other_job_inputs_do_not_consume_retry_limit(monkeypatch):
    monkeypatch.setenv('CRYOAGENT_UNATTENDED', '1')
    result = build_retry_candidate(failed(), {'nodes': [failed('J3', 'J9') for _ in range(10)] + [failed()]})
    assert result['available']


def test_request_budget_survives_restarts_and_checks_identity(tmp_path):
    for _ in range(3):
        assert unattended.consume_request(tmp_path, 'P1', 'W1', 'J1', {'requested_inputs': ['image']})
    assert not unattended.consume_request(tmp_path, 'P1', 'W1', 'J1', {})
    assert unattended.consume_request(tmp_path, 'P1', 'W1', 'J2', {})
    with pytest.raises(ValueError, match='another workspace'):
        unattended.consume_request(tmp_path, 'P1', 'W2', 'J2', {})


def image(role, uid):
    return dict(role=role, data_url='data:image/png;base64,' + str(uid))


def test_pilot_evidence_uses_same_calibrated_micrographs():
    calls = []
    async def call(session, name, args):
        calls.append((name, args))
        if name == 'get_pick_inspection_visual_context':
            return dict(panels=[{'micrograph_uid': 10}, {'micrograph_uid': 12}],
                evidence_images=[image('full_micrograph_pick_overlay', 1)],
                score_scatter=image('scatter', 3))
        assert args['micrograph_uids'] == ['10', '12']
        return dict(evidence_images=[image('full_micrograph_with_scale', 2)])
    def facts(*args):
        return dict(inspect_job_uid='J2', inspect_parameters={'ncc_score_thresh': .1})
    candidate = {'candidate_actions': [{'diameter_trial': {'role': 'accept_full', 'picker_uid': 'J1'}}]}
    result = asyncio.run(collect_current_pilot(call, None, 'P1', 'W1', 'J4', candidate, facts_loader=facts))
    assert result['success']
    assert len(calls) == 2
    assert len(result['attachments']) == 3
    message = message_for(result)
    assert 'ncc_score_thresh' in message['content'][0]['text']
    assert 'data:image' not in message['content'][0]['text']


def test_missing_pilot_image_does_not_invent_evidence():
    async def call(*args):
        return {'success': False, 'panels': []}
    result = asyncio.run(collect_current_pilot(call, None, 'P1', 'W1', 'J4',
        {'candidate_actions': [{'diameter_trial': {'role': 'retry', 'picker_uid': 'J1'}}]},
        facts_loader=lambda *args: {'inspect_job_uid': 'J2'}))
    assert result['success'] is False
    assert not result['attachments']


def test_requested_job_ids_are_workspace_scoped(monkeypatch):
    import sys
    class Job:
        def __init__(self, uid, parents):
            self.uid = uid
            self.model = SimpleNamespace(model_dump=lambda: dict(workspace_uids=['W1'], parents=parents,
                status='completed', spec={'type': 'class_2D_new', 'params': {}}))
    project = SimpleNamespace(find_workspace=lambda _: SimpleNamespace(find_jobs=lambda: [Job('J1', []), Job('J2', ['J1'])]))
    client = SimpleNamespace(find_project=lambda _: project)
    monkeypatch.setitem(sys.modules, 'cryosparc_client', SimpleNamespace(cryosparc_client=lambda: client))
    result = unattended.workspace_facts('P1', 'W1', 'J2', ['Get J999 and J1'])
    assert result['rejected_job_ids'] == ['J999']
    assert {r['uid'] for r in result['records']} == {'J1', 'J2'}


def test_gather_only_readonly_allowlisted_tools():
    calls = []
    async def call(session, name, args):
        calls.append(name)
        return {'contact_sheet': image('classes', 1)}
    facts = lambda *args: {'records': [{'uid': 'J1', 'job_type': 'class_2D_new'}]}
    result = asyncio.run(unattended.gather(call, None, 'P1', 'W1', 'J1', {},
        requested=['image'], facts_loader=facts))
    assert calls == ['get_class_average_visual_context']
    assert result['attachments']
    assert 'data:image' not in message_for(result)['content'][0]['text']
