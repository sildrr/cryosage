from types import SimpleNamespace
from unittest.mock import patch

import pytest

from inspect_score_contract import raw_candidate, raw_parameters
from job_specs import get_parameter_template
from dynamic_candidates import build_registry_candidate
from job_executor import execute_job_action, plan_job_action
from action_registry import validate_parameters


def test_static_template_supplies_raw_scale_and_rejects_calibration():
    template = get_parameter_template('inspect_picks_v2')
    errors, parameters = validate_parameters({'ncc_score_thresh': .2,'lpower_thresh_max':100}, template, 'parameters')
    assert not errors
    assert parameters['calibrate_ncc'] is False
    assert parameters['calibrate_pow'] is False
    errors, _ = validate_parameters({'calibrate_pow': True}, template, 'parameters')
    assert errors


def test_native_true_defaults_cannot_override_raw_contract():
    spec = SimpleNamespace(type='inspect_picks_v2', params={
        k: SimpleNamespace(default=True, type='boolean')
        for k in ('calibrate_ncc', 'calibrate_pow')})
    candidate = build_registry_candidate(dict(cryosparc_job_uid='J1', workflow_node_id='J1'), spec, {})
    for key in ('calibrate_ncc', 'calibrate_pow'):
        assert candidate['default_parameters'][key] is False
        assert candidate['parameter_template'][key]['const'] is False


def test_historical_candidate_is_copied_and_raw_defaults_override_old_true():
    old = dict(job_type='inspect_picks_v2', default_parameters={'calibrate_pow': True,'lpower_thresh_max':99999})
    candidate = raw_candidate(old)
    assert old['default_parameters']['calibrate_pow'] is True
    assert candidate['default_parameters']['calibrate_pow'] is False
    assert candidate['score_space'] == 'raw_pick_stats'
    assert 'lpower_thresh_max' not in candidate['default_parameters']
    other = dict(job_type='homo_refine_new')
    assert raw_candidate(other) is other


@pytest.mark.parametrize('key', ['calibrate_ncc', 'calibrate_pow'])
def test_execution_rejects_incompatible_calibration_before_contacting_server(key):
    with patch('job_executor.cryosparc_client') as client:
        with pytest.raises(ValueError, match='must be false'):
            execute_job_action('P6','W1',dict(job_type='inspect_picks_v2',
                resolved_parameters={key:True,'lpower_thresh_max':100}), dry_run=False)
        client.assert_not_called()


def test_dry_run_includes_actual_explicit_flags_without_mutating_action():
    action = dict(job_type='inspect_picks_v2', resolved_parameters={'lpower_thresh_min':19,'lpower_thresh_max':100})
    result = execute_job_action('P6','W1',action)
    assert result['planned_action']['resolved_parameters'] == dict(
        lpower_thresh_min=19,lpower_thresh_max=100,calibrate_ncc=False,calibrate_pow=False)
    assert action['resolved_parameters'] == {'lpower_thresh_min':19,'lpower_thresh_max':100}
    with pytest.raises(ValueError,match='lpower_thresh_max'):
        raw_parameters({})


def test_plan_also_records_explicit_raw_flags():
    action = dict(job_type='inspect_picks_v2', resolved_parameters={'ncc_score_thresh':.28,'lpower_thresh_max':100},
        plan_step=1, action_id='inspect', action_type='forward',workflow_node_id='J1',
        execution_mode='create_job')
    plan = plan_job_action(action)
    assert plan['resolved_parameters']['calibrate_ncc'] is False
    assert plan['resolved_parameters']['calibrate_pow'] is False
    assert action['resolved_parameters'] == {'ncc_score_thresh':.28,'lpower_thresh_max':100}


@pytest.mark.parametrize('value',[None,True,float('inf'),float('nan'),'100'])
def test_missing_or_nonfinite_upper_threshold_is_rejected(value):
    with pytest.raises(ValueError,match='lpower_thresh_max'):
        raw_parameters({'lpower_thresh_max':value})


@pytest.mark.parametrize('minimum',[100,101,float('nan'),True])
def test_inverted_or_invalid_power_range_is_rejected(minimum):
    with pytest.raises(ValueError,match='lpower_thresh_min'):
        raw_parameters({'lpower_thresh_min':minimum,'lpower_thresh_max':100})


def test_each_inspect_action_and_alias_requires_its_own_explicit_upper():
    from cryosparc_mcp_server import apply_inspect_parameter_guard
    valid=dict(success=True,issues=[])
    assert not apply_inspect_parameter_guard(dict(action='inspect_picks_v2',parameters={'ncc_score_thresh':.1}),valid)['success']
    both=dict(selected_actions=[dict(job_type='inspect_picks_v2',parameters={'lpower_thresh_max':100}),
                                dict(job_type='inspect_picks_v2',parameters={'ncc_score_thresh':.1})])
    assert not apply_inspect_parameter_guard(both,valid)['success']
    both['selected_actions'][1]['parameters']['lpower_thresh_max']=120
    assert apply_inspect_parameter_guard(both,valid)['success']
