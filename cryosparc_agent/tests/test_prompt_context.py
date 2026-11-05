from copy import deepcopy
import json

from prompt_context import compact_decision_payload, focus_class_attachments, coalesce_adjacent_user_messages


def test_compact_keeps_legal_schemas_and_errors_without_input_mutation():
    action = dict(action_id='select', job_type='select_2D', available=True,
                  parameter_template={'selected_templates': {'required': True}},
                  required_inputs={'templates': [{'source_job_uid': 'J5'}]})
    blocked = [dict(action_id=str(i), job_type='bad', blocked_by=['pilot restriction'],
                    parameter_template={'unused': 'x' * 20000}) for i in range(5)]
    context = dict(candidate_actions=[action], blocked_actions=blocked, project_uid='P6')
    failure = dict(execution_error={'message': 'Invalid selected_templates'}, candidate_actions=context)
    model = dict(current_state={'last_node_id': 'J5'}, candidate_actions=[action],
                 candidate_context=context, blocked_actions=blocked, failure_context=failure)
    original = deepcopy((model, context))
    payload = compact_decision_payload(model, context, 3, None)
    assert payload['candidate_actions_from_mcp']['candidate_actions'] == [action]
    summary = payload['candidate_actions_from_mcp']['blocked_actions_summary']
    assert summary['count'] == 5
    assert len(summary['groups']) == 1
    assert payload['failure_context']['execution_error']['message'] == 'Invalid selected_templates'
    assert 'candidate_actions' not in payload['model_input']
    assert 'candidate_actions' not in payload['failure_context']
    assert len(json.dumps(payload)) < 4000
    assert (model, context) == original


def test_class_sheet_moved_last_exact_bytes_no_duplicate_or_other_image_loss():
    image = dict(type='image_url', image_url={'url': 'data:image/png;base64,CLASS'})
    micrograph = dict(type='image_url', image_url={'url': 'data:image/png;base64,MIC'})
    messages = [dict(role='user', content=[{'type': 'text', 'text': 'class metadata'}, image]),
                dict(role='user', content=[micrograph])]
    original = deepcopy(messages)
    context = dict(kind='class_average', source={'job_uid': 'J5'},
                   contact_sheet={'data_url': image['image_url']['url']})
    result = focus_class_attachments(messages, context)
    assert result[-1]['content'][-1] == image
    images = [p for m in result for p in m.get('content', []) if p.get('type') == 'image_url']
    assert images == [micrograph, image]
    assert messages == original


def test_nonclass_prompt_unchanged_and_missing_sheet_not_invented():
    messages = [dict(role='user', content='unchanged')]
    assert focus_class_attachments(messages, {'kind': 'micrograph_preview'}) is messages
    assert focus_class_attachments(messages, {'kind': 'class_average'}) is messages


def test_adjacent_user_coalescing_preserves_pixels_and_text_order():
    image = dict(type='image_url', image_url={'url': 'data:image/png;base64,UNCHANGED'})
    messages = [dict(role='system', content='policy'),
                dict(role='user', content=[image]), dict(role='user', content='question')]
    original = deepcopy(messages)
    result = coalesce_adjacent_user_messages(messages)
    assert len(result) == 2
    assert result[1]['content'] == [image, {'type': 'text', 'text': 'question'}]
    assert messages == original


def test_coalescing_never_crosses_tools_or_assistant_turns():
    messages = [dict(role='user', content='evidence'),
                dict(role='assistant', content=None, tool_calls=[{'id': 'call1'}]),
                dict(role='tool', content='result', tool_call_id='call1'),
                dict(role='user', content='question')]
    assert coalesce_adjacent_user_messages(messages) == messages
