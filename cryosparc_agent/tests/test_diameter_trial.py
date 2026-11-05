import copy
import json

from action_registry import apply_hard_workflow_gates, validate_parameters
from autonomous_mcp_closed_loop import normalize_visual_sft_decision, build_autonomous_prompt
from diameter_trial import initial_trial, trial_candidates, MAX_ATTEMPTS
from visual_decision_contract import build_stage_decision_messages


def node(uid, kind, parents=(), title='', params=None):
    return dict(cryosparc_job_uid=uid, workflow_node_id=uid, logical_node_id=uid,
                job_type=kind, status='completed', parent_job_uids=list(parents),
                child_job_uids=[], title=title, key_parameters=params or {},
                inputs={}, outputs={})


def fixture():
    ctf = node('J1', 'patch_ctf_estimation_multi')
    ctf['outputs']['exposures'] = dict(available=True, num_items=200,
                                         result_names=['micrograph_blob', 'ctf'])
    picker = node('J2', 'blob_picker_gpu', ['J1'], 'Agent diameter_trial_forward_J1_blob_picker_gpu',
                  dict(diameter=80, diameter_max=140, num_process=20))
    picker['inputs']['micrographs'] = [dict(source_job_uid='J1', source_output='exposures')]
    inspect = node('J3', 'inspect_picks_v2', ['J2'])
    extract = node('J4', 'extract_micrographs_multi', ['J3'])
    classes = node('J5', 'class_2D_new', ['J4'])
    return dict(nodes=[ctf, picker, inspect, extract, classes]), picker, inspect, extract, classes


def action(uid='J3', kind='extract_micrographs_multi'):
    return dict(action_id='forward_' + uid + '_' + kind, action_type='forward',
                job_type=kind, available=True, required_inputs={
                    'particles': [dict(source_job_uid=uid, source_output='particles')]},
                parameter_template={}, default_parameters={})


def test_trial_caps_micrographs_and_does_not_mutate_original():
    full = action('J1', 'blob_picker_gpu')
    pilot = initial_trial(full, 200)
    assert pilot['default_parameters']['num_process'] == 20
    assert pilot['parameter_template']['num_process']['const'] == 20
    assert full['default_parameters'] == {}


def test_ctf_offers_trial_and_disable_keeps_original_menu(monkeypatch):
    monkeypatch.setenv('CRYOAGENT_FIXED_INITIAL_FUNNEL', '1')
    monkeypatch.setenv('CRYOAGENT_INITIAL_SCALE_CALIBRATION', '0')
    monkeypatch.setenv('CRYOAGENT_UNCERTAIN_DIAMETER_TRIAL', '1')
    state, *_ = fixture()
    ctf = state['nodes'][0]
    choices, _ = apply_hard_workflow_gates([action('J1', 'blob_picker_gpu')], [], state, ctf)
    assert len(choices) == 2
    assert choices[0]['parameter_template']['num_process']['const'] == 200
    assert choices[1]['parameter_template']['num_process']['const'] == 20
    historical = action('J99', 'blob_picker_gpu')
    choices, _ = apply_hard_workflow_gates(
        [action('J1', 'blob_picker_gpu'), historical], [], state, ctf)
    pilots = [a for a in choices if a.get('diameter_trial')]
    assert len(pilots) == 1
    assert pilots[0]['action_id'] == 'diameter_trial_forward_J1_blob_picker_gpu'
    monkeypatch.setenv('CRYOAGENT_UNCERTAIN_DIAMETER_TRIAL', '0')
    choices, _ = apply_hard_workflow_gates([action('J1', 'blob_picker_gpu')], [], state, ctf)
    assert len(choices) == 1


def test_uncertain_and_legacy_choose_small_trial_confident_keeps_full():
    full = action('J1', 'blob_picker_gpu')
    full['default_parameters']['num_process'] = 200
    candidates = dict(candidate_actions=[full, initial_trial(full, 200)])
    for confidence, count in [('uncertain', 20), ('confident', 200), (None, 20)]:
        decision = dict(decision='blob_picker_gpu', parameters=dict(diameter=80, diameter_max=140))
        if confidence:
            decision['diameter_confidence'] = confidence
        result = normalize_visual_sft_decision(decision, candidates, 'J1', {}, dict(micrograph_count=200))
        assert result['parameters']['num_process'] == count
        assert result['action_id'].startswith('diameter_trial_') == (count == 20)


def test_inspect_can_retry_or_continue_bounded_extraction():
    state, _, inspect, _, _ = fixture()
    candidates = trial_candidates(state, inspect, [action()])
    assert {a['diameter_trial']['role'] for a in candidates} == {'continue', 'retry'}
    continuation = candidates[0]
    assert continuation['default_parameters']['num_extract'] == 20
    result = normalize_visual_sft_decision(
        dict(decision='blob_picker_gpu', action_id=candidates[1]['action_id'],
             parameters=dict(diameter=120, diameter_max=180)),
        dict(candidate_actions=candidates), 'J3', {}, dict(micrograph_count=200))
    assert result['parameters']['num_process'] == 20


def test_class2d_review_allows_template_selection_and_full_reuses_original_ctf():
    state, _, _, _, classes = fixture()
    result = trial_candidates(state, classes, [action('J5', 'select_2D')])
    assert {a['diameter_trial']['role'] for a in result} == {'accept_full', 'retry', 'select_templates'}
    full = result[0]
    assert full['required_inputs']['micrographs'][0]['source_job_uid'] == 'J1'
    assert full['default_parameters'] == dict(diameter=80, diameter_max=140, num_process=200)
    assert full['parameter_template']['diameter']['const'] == 80
    assert full['parameter_template']['diameter_max']['const'] == 140
    errors, resolved = validate_parameters({}, full['parameter_template'], 'parameters')
    assert not errors
    assert resolved['diameter'] == 80
    assert resolved['diameter_max'] == 140
    assert resolved['num_process'] == 200
    errors, _ = validate_parameters({'diameter': 40}, full['parameter_template'], 'parameters')
    assert errors[0].code == 'parameter_const_mismatch'
    context = dict(candidate_actions=result)
    messages = build_autonomous_prompt({}, context, 1,
        dict(kind='class_average', class_statistics=[], contact_sheet={}), kb_tool_policy='disabled')
    text = json.dumps(messages)
    assert 'Do NOT return selected_templates' not in text
    assert 'solely to supply relatively better templates' in text
    assert 'small-sample particle-diameter trial' in text


def test_max_retries_requires_relative_promotion_instead_of_quality_stop():
    state, picker, _, _, classes = fixture()
    for i in range(1, MAX_ATTEMPTS):
        other = copy.deepcopy(picker)
        other.update(cryosparc_job_uid='J' + str(20 + i))
        state['nodes'].append(other)
    candidates = trial_candidates(state, classes, [])
    assert [a['diameter_trial']['role'] for a in candidates] == ['accept_full']
    assert 'must not cause stop/request_input' in candidates[0]['description']


def test_controller_specific_retry_extension(monkeypatch):
    state, picker, _, _, classes = fixture()
    for i in range(1, MAX_ATTEMPTS):
        other = copy.deepcopy(picker)
        other['cryosparc_job_uid'] = f'J{20 + i}'
        state['nodes'].append(other)
    monkeypatch.setenv('CRYOAGENT_DIAMETER_TRIAL_MAX_ATTEMPTS', '6')
    result = trial_candidates(state, classes, [])
    assert [a['diameter_trial']['role'] for a in result] == ['accept_full', 'retry']
    assert result[-1]['default_parameters']['num_process'] == 20
    for i in range(3):
        other = copy.deepcopy(picker)
        other['cryosparc_job_uid'] = f'J{30 + i}'
        state['nodes'].append(other)
    assert [a['diameter_trial']['role'] for a in trial_candidates(state, classes, [])] == ['accept_full']


def test_normal_branch_unchanged_and_missing_ctf_fails_closed():
    state, picker, _, _, classes = fixture()
    picker['title'] = 'Ordinary picker'
    assert trial_candidates(state, classes, []) is None
    picker['title'] = 'Agent diameter_trial_forward_J1_blob_picker_gpu'
    state['nodes'][0]['outputs'] = {}
    assert trial_candidates(state, classes, [action('J5', 'select_2D')]) == []


def test_pilot_gates_override_fixed_funnel_and_resume_from_dag():
    state, _, _, _, classes = fixture()
    choices, blocked = apply_hard_workflow_gates(
        [action('J5', 'select_2D')], [], state, classes)
    assert {c['job_type'] for c in choices} == {'blob_picker_gpu', 'select_2D'}
    assert not blocked


def test_selected_pilot_templates_only_feed_full_ctf_template_picker():
    state, picker, _, _, classes = fixture()
    selected = node('J6', 'select_2D', ['J5'])
    selected['outputs']['templates_selected'] = dict(
        available=True, num_items=2, result_names=['blob'])
    state['nodes'].append(selected)
    choices = trial_candidates(state, selected, [action('J6', 'homo_abinit')])
    assert len(choices) == 2
    assert choices[0]['job_type'] == 'template_picker_gpu'
    assert choices[0]['required_inputs']['templates'][0]['source_job_uid'] == 'J6'
    assert choices[0]['required_inputs']['micrographs'][0]['source_job_uid'] == 'J1'
    assert 'particles' not in choices[0]['required_inputs']
    assert choices[1]['diameter_trial']['role']=='accept_full'
    selected['outputs']['templates_selected']['num_items'] = 0
    assert trial_candidates(state, selected, []) == []
    selected['outputs'] = {'templates': dict(available=True, num_items=20)}
    assert trial_candidates(state, selected, []) == []


def test_template_descendants_do_not_reenter_pilot_through_template_ancestry():
    state, *_ = fixture()
    selected = node('J6', 'select_2D', ['J5'])
    picker = node('J7', 'template_picker_gpu', ['J1', 'J6'])
    inspect = node('J8', 'inspect_picks_v2', ['J7'])
    state['nodes'] += [selected, picker, inspect]
    assert trial_candidates(state, picker, []) is None
    assert trial_candidates(state, inspect, []) is None


def test_pilot_selection_cannot_use_another_class_branch():
    state, _, _, _, classes = fixture()
    candidates = trial_candidates(state, classes, [action('J99', 'select_2D')])
    assert all(c['job_type'] != 'select_2D' for c in candidates)


def test_full_picker_is_not_reinterpreted_as_pilot():
    state, picker, _, _, _ = fixture()
    picker['title'] = 'Agent diameter_full_J5'
    assert trial_candidates(state, picker, []) is None


def test_prompt_contract_exposes_uncertainty_without_changing_training_schema():
    context = dict(candidate_actions=[])
    live = json.dumps(build_stage_decision_messages('blob_picker_diameter', context, include_reason=True))
    train = json.dumps(build_stage_decision_messages('blob_picker_diameter', context))
    assert 'diameter_confidence' in live
    assert 'diameter_confidence' not in train


def test_extract_review_prompt_allows_diameter_rollback():
    state, _, inspect, _, _ = fixture()
    choices = trial_candidates(state, inspect, [action()])
    text = json.dumps(build_stage_decision_messages(
        'extract_box_size', dict(candidate_actions=choices), include_reason=True))
    assert 'action_id' in text
    assert 'retry' in text
