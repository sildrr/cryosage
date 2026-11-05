from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from dynamic_candidates import build_registry_candidate
from job_executor import plan_job_action
from job_executor import execute_job_action
from job_input_contracts import normalize_input_parameters, normalize_volume_connections
from quality_policy import build_retry_candidate
from workflow_state import extract_run_errors


def native_spec():
    return SimpleNamespace(type='class_3D', title='3D Classification', category='classification',
        stability='stable', tags=['gpuEnabled'], interactive=False, params={},
        inputs=SimpleNamespace(root={'particles': SimpleNamespace(count_min=1),
                                     'volume': SimpleNamespace(count_min=0, repeat_allowed=True)}))


def failed():
    return dict(cryosparc_job_uid='J3', workflow_node_id='J3', job_type='class_3D', status='failed',
        key_parameters={}, inputs={'particles': [dict(source_job_uid='J2', source_output='particles')],
                                  'volume': [dict(source_job_uid='J2', source_output='volume')]},
        run_errors={'worker_traceback': 'Input volumes detected, please set initialization mode to input'})


def test_volume_mode_is_canonicalized_without_changing_other_choices():
    params = dict(class3D_init_mode='simple', class3D_N_K=3, class3D_target_res=6.)
    normalized = normalize_input_parameters('class_3D', params, {'volume': ('J2', 'volume')})
    assert normalized == {**params, 'class3D_init_mode': 'input'}
    assert params['class3D_init_mode'] == 'simple'
    with pytest.raises(ValueError):
        normalize_input_parameters('class_3D', {'class3D_init_mode': 'input'}, {})


def test_candidate_exposes_input_initialization_contract():
    node = failed()
    candidate = build_registry_candidate(node, native_spec(), node['inputs'])
    assert candidate['default_parameters']['class3D_init_mode'] == 'input'
    assert candidate['parameter_template']['class3D_init_mode']['const'] == 'input'


def test_authorized_retry_preserves_native_flags_and_original_parameters(monkeypatch):
    monkeypatch.setenv('CRYOAGENT_UNATTENDED', '1')
    node = failed()
    params = dict(class3D_N_K=3, class3D_target_res=6., class3D_init_mode='simple', random_seed=123)
    candidate = build_retry_candidate(node, {'nodes': [node]}, native_spec(), params)
    action = dict(plan_step=1, action_id=candidate['action_id'], action_type='forward',
        workflow_node_id=candidate['workflow_node_id'], job_type='class_3D',
        execution_mode='create_job', resolved_parameters=candidate['default_parameters'])
    plan = plan_job_action(action, candidate, lane='test_lane')
    assert not plan['approval_required']
    assert plan['requires_gpu']
    assert plan['resolved_parameters'] == {**params, 'class3D_init_mode': 'input'}
    assert candidate['failure_diagnostics']['worker_traceback']
    assert len(candidate['required_inputs']['volume']) == 3
    assert plan['connections']['volume'] == [('J2', 'volume')] * 3
    monkeypatch.setenv('CRYOAGENT_UNATTENDED', '0')
    candidate = build_retry_candidate(node, {'nodes': [node]}, native_spec(), params)
    action['approval_required'] = candidate['approval_required']
    assert plan_job_action(action, candidate, lane='test_lane')['approval_required']


def test_failed_worker_traceback_is_available_even_without_api_errors(tmp_path):
    (tmp_path/'job.log').write_text('startup configuration\nTraceback (most recent call last):\nAssertionError: Input volumes detected\n')
    job = SimpleNamespace(status='failed', dir=tmp_path, model=SimpleNamespace())
    result = extract_run_errors(job)
    assert 'AssertionError: Input volumes detected' in result['worker_traceback']
    assert 'startup configuration' not in result['worker_traceback']
    job.status = 'completed'
    assert 'worker_traceback' not in extract_run_errors(job)


def test_volume_count_expansion_is_bounded_and_preserves_distinct_references():
    source = {'volume': [('J2', 'volume')], 'particles': ('J2', 'particles')}
    fixed = normalize_volume_connections('class_3D', {'class3D_N_K': 3}, source, repeat_allowed=True)
    assert fixed['volume'] == [('J2', 'volume')] * 3
    assert source['volume'] == [('J2', 'volume')]
    assert normalize_volume_connections('class_3D', {'class3D_N_K': 3}, fixed, repeat_allowed=True) == fixed
    with pytest.raises(ValueError):
        normalize_volume_connections('class_3D', {'class3D_N_K': 3}, source)
    with pytest.raises(ValueError):
        normalize_volume_connections('class_3D', {'class3D_N_K': 3},
            {'volume': [('J2', 'volume'), ('J3', 'volume')]}, repeat_allowed=True)
    distinct = {'volume': [('J2', 'volume'), ('J3', 'volume'), ('J4', 'volume')]}
    assert normalize_volume_connections('class_3D', {'class3D_N_K': 3}, distinct) == distinct
    with pytest.raises(ValueError):
        normalize_volume_connections('class_3D', {'class3D_N_K': 101}, source, repeat_allowed=True)


def test_live_executor_connects_three_initial_volumes_before_queue():
    job = Mock(uid='J4', status='queued')
    job.inputs = {'volume': SimpleNamespace(connections=[])}
    def connect(slot, uid, output):
        if slot == 'volume':
            job.inputs['volume'].connections.append((uid, output))
    job.connect.side_effect = connect
    workspace = Mock()
    workspace.create_job.return_value = job
    client = Mock()
    client.job_register.specs = [native_spec()]
    client.find_workspace.return_value = workspace
    plan = dict(job_type='class_3D', action_id='retry_J3', approval_required=False,
        resolved_parameters={'class3D_N_K': 3, 'class3D_init_mode': 'input'},
        connections={'particles': ('J2', 'particles'), 'volume': ('J2', 'volume')},
        queue=dict(will_queue=True, lane='test', gpus=[], cluster_vars={}, queue_without_lane=False))
    with patch('job_executor.cryosparc_client', return_value=client):
        result = execute_job_action('P6', 'W1', plan, dry_run=False)
    assert result['success']
    assert job.inputs['volume'].connections == [('J2', 'volume')] * 3
    job.queue.assert_called_once()
