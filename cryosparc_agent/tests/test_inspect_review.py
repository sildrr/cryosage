import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from inspect_review import membership, review_candidate, threshold_error, INSTRUCTIONS


def fixture():
    picker = dict(cryosparc_job_uid='J1', workflow_node_id='J1', status='completed',
        job_type='template_picker_gpu', inputs={}, outputs={
            'particles': {'available': True}, 'micrographs': {'available': True}})
    inspect = dict(cryosparc_job_uid='J2', workflow_node_id='J2', status='completed',
        job_type='inspect_picks_v2', outputs={}, inputs={
            name: [dict(source_job_uid='J1', source_output=name)] for name in ('particles', 'micrographs')},
        key_parameters=dict(ncc_score_thresh=.2, lpower_thresh_min=750., lpower_thresh_max=1300.,
                            calibrate_ncc=False, calibrate_pow=False, do_auto_cluster=False))
    return {'nodes': [picker, inspect]}, inspect, [dict(job_type='extract_micrographs_multi', available=True)]


def test_revision_reuses_raw_picker_not_already_accepted_subset():
    s, i, actions = fixture()
    c = review_candidate(s, i, actions)
    assert c['required_inputs']['particles'][0]['source_job_uid'] == 'J1'
    assert c['action_id'] == 'inspect_review_J2'
    assert c['execution_mode'] == 'create_job'
    assert c['inspect_review']['revision'] == 1
    assert c['parameter_template']['calibrate_ncc']['const'] is False


def test_revision_limit_survives_restart_via_native_jobs():
    s, i, actions = fixture()
    for uid in ('J3', 'J4'):
        other = copy.deepcopy(i)
        other['cryosparc_job_uid'] = other['workflow_node_id'] = uid
        s['nodes'].append(other)
    assert review_candidate(s, s['nodes'][-1], actions) is None
    assert review_candidate(s, s['nodes'][-1], actions) is None


def test_active_revision_prevents_duplicate_and_no_downstream_means_no_review():
    s, i, actions = fixture()
    active = copy.deepcopy(i)
    active.update(cryosparc_job_uid='J3', workflow_node_id='J3', status='running')
    s['nodes'].append(active)
    assert review_candidate(s, i, actions) is None
    assert review_candidate(s, i, []) is None


def test_noop_inverted_or_nonfinite_bounds_rejected_but_relaxation_allowed():
    s, i, actions = fixture()
    c = review_candidate(s, i, actions)
    assert threshold_error({}, c)
    assert threshold_error({'lpower_thresh_min': float('nan')}, c)
    assert threshold_error({'lpower_thresh_max': 500}, c)
    assert threshold_error({'calibrate_pow': True}, c)
    assert threshold_error({'lpower_thresh_min': 550}, c) is None
    assert threshold_error({'lpower_thresh_max': 1800}, c) is None


def test_validation_refuses_switching_to_the_filtered_subset():
    from schemas import Action
    from action_registry import validate_action_against_candidates
    s, i, actions = fixture()
    c = review_candidate(s, i, actions)
    action = Action(action_id=c['action_id'], action_type='forward', workflow_node_id='J2',
        job_type='inspect_picks_v2', parameters={'lpower_thresh_min': 550},
        connections={'particles': {'source_job_uid': 'J2', 'source_output': 'particles'}})
    issues, _, _ = validate_action_against_candidates(action, {c['action_id']: c}, 0)
    assert 'inspect_review_inputs_changed' in {issue.code for issue in issues}


def test_matching_uses_actual_uid_not_score_threshold_or_row_position():
    assert membership([50, 3, 8, 10], [10, 3]).tolist() == [False, True, False, True]
    with pytest.raises(ValueError, match='not a subset'):
        membership([50, 3], [5])
    with pytest.raises(ValueError, match='unique'):
        membership([3, 3], [3])


def test_prompt_attaches_clean_and_membership_images_without_base64_in_metadata():
    from autonomous_mcp_closed_loop import build_autonomous_prompt, select_execution_api_model
    urls = ['data:image/png;base64,clean', 'data:image/png;base64,membership']
    visual = dict(kind='inspect_result_review', input_count=1000, accepted_count=300,
        evidence_images=[dict(data_url=url, description='same micrograph') for url in urls],
        score_scatter=dict(data_url='data:image/png;base64,scatter'))
    messages = build_autonomous_prompt({'current_state': {}, 'dataset_info': {}},
        {'candidate_actions': []}, 1, visual_context=visual, kb_tool_policy='disabled')
    parts = messages[-1]['content']
    images = [p['image_url']['url'] for p in parts if p['type'] == 'image_url']
    assert all(isinstance(url, str) and url.startswith('data:image/') for url in images)
    assert len(images) == 3
    assert set(urls).issubset(images)
    texts = '\n'.join(p['text'] for p in parts if p['type'] == 'text')
    assert 'COMPLETED Inspect Picks' in texts
    assert 'base64,clean' not in texts
    args = SimpleNamespace(api_model='strong', fixed_api_model='cheap')
    assert select_execution_api_model(args, {'inspect_review_required': True}) == 'strong'


def test_mcp_exposes_review_tool():
    import asyncio
    from cryosparc_mcp_server import mcp
    tools = asyncio.run(mcp.list_tools())
    assert 'get_inspect_pick_review_context' in {t.name for t in tools}
