"""Compact model-facing context without changing validation or execution data."""
from collections import defaultdict
from copy import deepcopy
import json


def coalesce_adjacent_user_messages(messages):
    """Keep multimodal evidence in one turn for gateways reading only the last user."""
    result = []
    for message in deepcopy(messages):
        if (message.get('role') == 'user' and result and result[-1].get('role') == 'user'
                and set(message) <= {'role', 'content'} and set(result[-1]) <= {'role', 'content'}):
            previous = result[-1].get('content')
            current = message.get('content')
            if isinstance(previous, str):
                previous = [dict(type='text', text=previous)]
            if isinstance(current, str):
                current = [dict(type='text', text=current)]
            if isinstance(previous, list) and isinstance(current, list):
                result[-1]['content'] = previous + current
                continue
        result.append(message)
    return result


def compact_blocked_actions(actions):
    groups = defaultdict(list)
    for action in actions or []:
        if not isinstance(action, dict):
            continue
        reasons = action.get('blocked_by') or action.get('missing_required_inputs') or ['Unavailable']
        groups[json.dumps(reasons, sort_keys=True, default=str)].append({
            'action_id': action.get('action_id'), 'job_type': action.get('job_type')})
    return dict(count=sum(len(items) for items in groups.values()),
                groups=[dict(blocked_by=json.loads(reason), actions=items)
                        for reason, items in groups.items()],
                notice='Unavailable actions are diagnostic context, not executable choices. Full schemas remain in candidate_actions.json.')


def compact_failure_context(failure):
    if not isinstance(failure, dict):
        return deepcopy(failure)
    result = deepcopy(failure)
    if 'candidate_actions' in result:
        result.pop('candidate_actions')
        result['candidate_actions_reference'] = 'Use the current candidate_actions_from_mcp; historical full schemas omitted.'
    return result


def compact_decision_payload(model_input, candidate_context, round_index, symmetry_evidence):
    context = deepcopy(candidate_context)
    if 'blocked_actions' in context:
        context['blocked_actions_summary'] = compact_blocked_actions(context.pop('blocked_actions'))
    state = deepcopy(model_input)
    # Full executable schemas occur once, at the outer authoritative context.
    for key in ('candidate_actions', 'blocked_actions', 'candidate_context', 'failure_context'):
        state.pop(key, None)
    return dict(round=round_index, model_input=state,
        candidate_actions_from_mcp=context,
        decision_guidance=context.get('workflow_guidance') or {},
        failure_context=compact_failure_context(model_input.get('failure_context')),
        symmetry_evidence=deepcopy(symmetry_evidence))


def focus_class_attachments(messages, visual_context):
    """Move existing class sheets last, with source labels; do not add duplicates."""
    if not visual_context:
        return messages
    kind = visual_context.get('kind')
    contexts = (visual_context.get('contexts') or []) if kind == 'class_average_comparison' else [visual_context]
    if kind not in {'class_average', 'class_average_comparison'}:
        return messages
    sheets = []
    for item in contexts:
        sheet = item.get('contact_sheet') or {}
        if sheet.get('data_url'):
            sheets.append((sheet['data_url'], item.get('source') or {}))
    if not sheets:
        return messages
    result = deepcopy(messages)
    moved = []
    seen = set()
    sources = dict(sheets)
    for message in result:
        if not isinstance(message.get('content'), list):
            continue
        kept = []
        for part in message['content']:
            url = (part.get('image_url') or {}).get('url') if part.get('type') == 'image_url' else None
            if url in sources:
                if url not in seen:
                    seen.add(url)
                    moved.extend([dict(type='text', text='Class 2D contact sheet source: ' + json.dumps(sources[url])), part])
            else:
                kept.append(part)
        message['content'] = kept
    if moved:
        result.append(dict(role='user', content=[dict(type='text', text=(
            'The following attachments are the labelled Class 2D averages, not micrographs. '
            'Inspect their class_id labels and source-scoped CryoSift scores before selecting templates. '
            'Distinguish an unreadable panel or uncertain quality from an absent attachment. '
            'Lower CryoSift scores are better; do not infer absence from an empty recommendation. '
            'Report only observations you can substantiate.')), *moved]))
    return result
