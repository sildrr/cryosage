from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from v2_decision_adapter import normalize_requested_actions, adapt_v2_decision_to_internal
from experiment_planner import decision_respects_plan
from job_executor import plan_job_action, execute_job_action, build_explicit_connections
from autonomous_mcp_closed_loop import find_created_jobs


def candidate():
    return dict(action_id='select_J5', job_type='select_2D', action_type='forward',
        workflow_node_id='J5:select_2D', available=True,
        parameter_template={'selected_templates': {'type': 'string'}},
        default_parameters={}, required_inputs={'templates': [dict(source_job_uid='J5', source_output='class_averages')]})


@pytest.mark.parametrize('shape', ['nested', 'id_only', 'string_id', 'selected'])
def test_action_shapes_match_same_scoped_candidate(shape):
    action = dict(action_id='select_J5', job_type='select_2D', parameters={'selected_templates': '1,3'})
    d = dict(decision_type='forward', reason='test')
    if shape == 'nested':
        d['action'] = action
    elif shape == 'id_only':
        d.update(action_id=action['action_id'], parameters=action['parameters'])
    elif shape == 'string_id':
        d.update(action=action['action_id'], parameters=action['parameters'])
    else:
        d['selected_actions'] = [action]
    assert decision_respects_plan(d, {'approved_action_ids': ['select_J5']})[0]
    result = adapt_v2_decision_to_internal(d, [candidate()], 'J5')
    assert result['success']
    assert result['internal_decision']['selected_actions'][0]['job_type'] == 'select_2D'


def test_conflicting_nested_fields_and_outside_plan_are_not_repaired():
    assert not normalize_requested_actions(dict(action={'action_id': 'a'}, action_id='b'))
    assert not decision_respects_plan(dict(action={'action_id': 'outside'}), {'approved_action_ids': ['allowed']})[0]


def test_empty_connection_uses_candidate_but_malformed_explicit_connection_is_rejected():
    action = dict(job_type='select_2D', resolved_parameters={'selected_templates': '1'}, connections={},
                  plan_step=1, action_id='select_J5', action_type='forward', workflow_node_id='J5', execution_mode='create_job')
    original = deepcopy(action)
    plan = plan_job_action(action, candidate())
    assert plan['connections'] == {'templates': ('J5', 'class_averages')}
    assert action == original
    with pytest.raises(ValueError):
        build_explicit_connections({'templates': []})
    with pytest.raises(ValueError, match='Missing required'):
        plan_job_action({**action, 'connections': {'particles': ('J5', 'particles')}}, candidate())


def test_live_required_inputs_checked_before_native_job_creation():
    cs = Mock()
    cs.job_register.specs = []
    plan = dict(job_type='homo_refine_new', connections={}, resolved_parameters={},
                approval_required=False, queue={'will_queue': True, 'lane': 'lane'})
    with patch('job_executor.cryosparc_client', return_value=cs):
        result = execute_job_action('P6', 'W119', plan, dry_run=False)
    assert not result['success']
    assert result['job_uid'] is None
    assert 'Missing required input' in result['error']
    cs.find_workspace.assert_not_called()


def test_failed_or_unqueued_job_never_enters_pending_list():
    items = [dict(job_uid='J1', success=False, status='failed'),
             dict(job_uid='J2', success=True, status='building', queued=False),
             dict(job_uid='J3', success=True, status='queued', queued=True),
             dict(job_uid='J4', success=True, status='completed')]
    result = find_created_jobs(dict(execution_result={'execution_results': items}))
    assert [job['job_uid'] for job in result] == ['J3', 'J4']
