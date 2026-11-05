import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import autonomous_mcp_closed_loop as runner
from experiment_planner import build_policy_plan, validate_plan
from reconstruction_observer import execute_observation_tasks


def task(**arguments):
    return dict(task_id='views', tool='get_abinitio_class2d_template_matching_context',
                arguments=arguments or {'abinitio_job_uid':'J1'})


def template(allowed=True, available=True, gap=.4):
    return dict(action_id='template',job_type='create_templates',available=available,
        refinement_rollback=True,rollback_kind='template_repick',
        template_repick_gate=dict(allowed=allowed,target_gap_A=gap))


def reselect():
    return dict(action_id='select',job_type='select_2D',available=True,
                refinement_rollback=True,rollback_kind='reselect_2d')


def test_request_input_with_tasks_is_canonicalized_without_mutation():
    raw=dict(decision_type='request_input', observation_tasks=[task()], approved_action_ids=[])
    original=copy.deepcopy(raw)
    result=validate_plan(raw,[],1)
    assert result['decision_type']=='request_observations'
    assert result['observation_tasks'][0]['arguments']=={'job_uid':'J1'}
    assert result['request_normalization']=='request_input_with_tasks_to_request_observations'
    assert raw==original


def test_genuine_missing_input_without_tasks_is_not_reinterpreted():
    result=validate_plan({'decision_type':'request_input','observation_tasks':None},[],1)
    assert result['decision_type']=='request_input'


@pytest.mark.parametrize('tasks', [[task()]*5, [{'tool':'execute_job','arguments':{}}], 'invalid'])
def test_compatibility_never_accepts_unsafe_or_unbounded_requests(tasks):
    with pytest.raises(ValueError):
        validate_plan(dict(decision_type='request_input',observation_tasks=tasks),[],1)


def test_observation_and_job_submission_cannot_be_combined():
    with pytest.raises(ValueError,match='cannot also approve'):
        validate_plan(dict(decision_type='request_input',observation_tasks=[task()],
                           approved_action_ids=['select']),[reselect()],1)


def test_alias_conflicts_are_rejected():
    with pytest.raises(ValueError,match='conflicting'):
        validate_plan(dict(decision_type='request_input',observation_tasks=[
            task(job_uid='J1',abinitio_job_uid='J2')]),[],1)


def state():
    return dict(project_uid='P6',workspace_uid='W93',nodes=[dict(
        cryosparc_job_uid='J1',workflow_node_id='J1',job_type='homo_abinit',status='completed',
        outputs={},inputs={},parent_job_uids=[],child_job_uids=[],run_errors={})])


def test_observer_executes_alias_in_the_assigned_workspace():
    with patch('reconstruction_observer.build_template_matching_consistency_context',return_value={'success':True}) as matcher:
        result=execute_observation_tasks(state(),{},[
            task(project_uid='P6',workspace_uid='W93',abinitio_job_uid='J1')])
    matcher.assert_called_once_with('P6','J1',30)
    assert result[0]['arguments']=={'job_uid':'J1'}


@pytest.mark.parametrize('arguments',[
    {'project_uid':'P99','abinitio_job_uid':'J1'},
    {'workspace_uid':'W99','abinitio_job_uid':'J1'},
    {'abinitio_job_uid':'J999'},
])
def test_observer_rejects_foreign_scope_or_job_before_tool_execution(arguments):
    with patch('reconstruction_observer.build_template_matching_consistency_context') as matcher:
        with pytest.raises(ValueError):
            execute_observation_tasks(state(),{},[task(**arguments)])
        matcher.assert_not_called()


def test_fallback_prefers_only_gated_available_template_over_reselection():
    candidates=[reselect(),template()]
    result=build_policy_plan({},candidates,[],1)
    assert result['approved_action_ids']==['template']
    for t in [template(allowed=False),template(available=False),
              {k:v for k,v in template().items() if k!='template_repick_gate'}]:
        assert build_policy_plan({},[reselect(),t],[],1)['approved_action_ids']==['select']


def test_near_target_one_time_nu_preference_remains_and_repetition_is_blocked():
    nu=dict(action_id='nu',job_type='nonuniform_refine_new',available=True,
        refinement_rollback=True,refinement_strategy_switch=True,
        required_inputs={'particles':[{'source_job_uid':'J1'}]})
    candidates=[reselect(),template(gap=.05),nu]
    assert build_policy_plan({},candidates,[],1)['approved_action_ids']==['nu']
    history=[dict(job_type=nu['job_type'],input_signature=json.dumps(nu['required_inputs'],sort_keys=True))]
    assert build_policy_plan({},candidates,history,1)['approved_action_ids']==['template']


def test_eligible_template_beats_empty_cryosift_fallback_but_missing_template_keeps_old_policy():
    inspect=dict(action_id='inspect',job_type='inspect_picks_v2',available=True,
                 refinement_rollback=True,rollback_kind='reinspect_picks')
    observation=dict(cryosift_evidence=[dict(success=True,selected_class_ids=[],class_scores=[{'score':4}])])
    assert build_policy_plan(observation,[reselect(),inspect,template()],[],1)['approved_action_ids']==['template']
    assert build_policy_plan(observation,[reselect(),inspect],[],1)['approved_action_ids']==['inspect']


def args():
    return SimpleNamespace(project='P6',workspace='W93',cryosift=False,planning_mode='api',
        api_key=None,api_key_env='OPENAI_API_KEY',api_base='https://example.invalid/v1',
        api_model='test',max_new_tokens=4096,temperature=0,stage_guidance=None)


def test_realistic_request_observe_then_plan_keeps_api_choice_without_fallback():
    request=dict(decision_type='request_input',observation_tasks=[
        task(project_uid='P6',abinitio_job_uid='J1')])
    final=dict(decision_type='plan_experiments',approved_action_ids=['template'])
    call=AsyncMock(side_effect=[{'dag':{'nodes':[]}}, {'success':True,'results':[{'result':'observed'}]}])
    with patch.object(runner,'call_tool_json',call), \
         patch.object(runner,'resolve_api_key',return_value='test-only'), \
         patch.object(runner,'run_openai_compatible_model',side_effect=[
             {'raw_text':json.dumps(request)},{'raw_text':json.dumps(final)}]) as api, \
         patch.object(runner,'build_policy_plan',wraps=build_policy_plan) as fallback:
        observation,plan,trace=asyncio.run(runner.build_round_plan(None,args(),{},
            {'candidate_actions':[reselect(),template()]},[]))
    assert plan['approved_action_ids']==['template']
    assert trace['fallback_used'] is False
    assert observation['planner_requested_evidence']==[{'result':'observed'}]
    assert call.call_args_list[1].args[1]=='execute_reconstruction_observation_tasks'
    assert call.call_args_list[1].args[2]['tasks'][0]['arguments']['job_uid']=='J1'
    assert api.call_count==2
    fallback.assert_not_called()


def test_actual_api_failure_uses_template_aware_fallback():
    with patch.object(runner,'call_tool_json',AsyncMock(return_value={})), \
         patch.object(runner,'resolve_api_key',return_value='test-only'), \
         patch.object(runner,'run_openai_compatible_model',side_effect=TimeoutError('test outage')):
        _,plan,trace=asyncio.run(runner.build_round_plan(None,args(),{},
            {'candidate_actions':[reselect(),template()]},[]))
    assert plan['approved_action_ids']==['template']
    assert trace['fallback_used']
    assert trace['planner_error']['error_type']=='TimeoutError'


def test_a_valid_api_reselection_is_not_replaced_by_fallback_template_preference():
    with patch.object(runner,'call_tool_json',AsyncMock(return_value={})), \
         patch.object(runner,'resolve_api_key',return_value='test-only'), \
         patch.object(runner,'run_openai_compatible_model',return_value={
             'raw_text':json.dumps(dict(decision_type='plan_experiments',approved_action_ids=['select']))}), \
         patch.object(runner,'build_policy_plan',wraps=build_policy_plan) as fallback:
        _,plan,trace=asyncio.run(runner.build_round_plan(None,args(),{},
            {'candidate_actions':[reselect(),template()]},[]))
    assert plan['approved_action_ids']==['select']
    assert not trace['fallback_used']
    fallback.assert_not_called()
