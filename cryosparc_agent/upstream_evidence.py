"""Stage-aware current-lineage observations for diameter-pilot reviews."""
import asyncio
import inspect
from pathlib import Path


def review_picker(context, current=None):
    candidates = context.get('candidate_actions', [])
    for candidate in candidates:
        recovery = candidate.get('pilot_template_recovery') or {}
        if recovery.get('pilot_picker'):
            return recovery['pilot_picker']
    # Promotion options can describe other completed pilots. Prefer the active
    # continuation/selection rather than whichever historical option comes first.
    for candidate in candidates:
        trial = candidate.get('diameter_trial') or {}
        if trial.get('role') in ('continue', 'retry', 'select_templates') and trial.get('picker_uid'):
            return trial['picker_uid']
    for candidate in candidates:
        trial = candidate.get('diameter_trial') or {}
        if (trial.get('role') == 'accept_full' and trial.get('picker_uid')
                and (current is None or trial.get('class_job_uid') in (None, current))):
            return trial['picker_uid']
    return None


def lineage_facts(project_uid, class_uid, workspace_uid, expected_picker):
    from cryosparc_client import cryosparc_client
    project = cryosparc_client().find_project(project_uid)
    pending, seen, jobs = [class_uid], set(), {}
    while pending and len(seen) < 16:
        uid = pending.pop(0)
        if uid in seen:
            continue
        seen.add(uid)
        job = project.find_job(uid)
        data = job.model.model_dump()
        if workspace_uid not in data.get('workspace_uids', []):
            raise ValueError('Evidence lineage leaves the assigned workspace: ' + uid)
        jobs[uid] = data
        if uid == expected_picker:
            continue
        pending.extend(data.get('parents', []))
    picker = jobs.get(expected_picker)
    if picker is None or picker['spec']['type'] not in ('blob_picker_gpu', 'template_picker_gpu'):
        raise ValueError('Candidate picker is not an ancestor of this pilot')
    current = jobs[class_uid]
    stage = current['spec']['type']
    if stage not in {'inspect_picks_v2', 'extract_micrographs_multi', 'class_2D_new', 'select_2D'}:
        raise ValueError('Unsupported pilot review stage: ' + stage)
    inspections = [d for d in jobs.values() if d['spec']['type'] == 'inspect_picks_v2']
    extracts = [d for d in jobs.values() if d['spec']['type'] == 'extract_micrographs_multi']
    expected_extracts = 0 if stage == 'inspect_picks_v2' else 1
    if len(inspections) != 1 or len(extracts) != expected_extracts:
        raise ValueError('Unexpected Inspect/Extract lineage for pilot review stage ' + stage)
    inspection = inspections[0]
    extraction = extracts[0] if extracts else None
    return dict(project_uid=project_uid, workspace_uid=workspace_uid,
        current_job_uid=class_uid, review_stage=stage,
        class2d_job_uid=(class_uid if stage == 'class_2D_new' else next(
            (d['uid'] for d in jobs.values() if d['spec']['type'] == 'class_2D_new'), None)),
        selected_templates=(current['spec'].get('params', {}).get('selected_templates')
                            if stage == 'select_2D' else None),
        picker_job_uid=expected_picker, inspect_job_uid=inspection['uid'],
        extract_job_uid=extraction['uid'] if extraction else None,
        picker_parameters={k:v for k,v in picker['spec'].get('params', {}).items()
                           if k in ('diameter', 'diameter_max', 'num_process')},
        inspect_parameters=inspection['spec'].get('params', {}),
        extract_parameters=({k:v for k,v in extraction['spec'].get('params', {}).items()
                             if k in ('box_size_pix', 'num_extract', 'bin_size_pix')}
                            if extraction else None))


def metadata(value):
    if isinstance(value, dict):
        return {k:metadata(v) for k,v in value.items()
                if k not in ('data_url', 'contact_sheet', 'evidence_images', 'score_scatter')}
    if isinstance(value, list):
        return [metadata(v) for v in value]
    return value


def full_images(context, role):
    candidates = [context.get('contact_sheet') or {}, *(context.get('evidence_images') or [])]
    images, urls = [], set()
    for item in candidates:
        url = item.get('data_url')
        if item.get('role') == role and isinstance(url, str) and url.startswith('data:image/') and url not in urls:
            images.append(dict(role=role, description=item.get('description'), data_url=url))
            urls.add(url)
    return images[:2]


def message_for(bundle):
    import json
    facts = dict(instruction=(
        'Additional read-only current-workspace observations, not ground truth or a '
        'recommended action. For diameter-pilot reviews compare the uncropped micrographs '
        'and calibrated scale bars, accepted-pick overlays, applied thresholds, and the '
        'pilot Class 2D images only when they have actually been supplied. Extract '
        'and Class 2D may not yet exist at an Inspect Picks review; their missing '
        'values are unknown, not failed measurements. Overlay colors are score percentiles, '
        'not validated particle-quality labels. Keep the original legal candidates, '
        'diameter retry limits and budget unchanged. Missing evidence remains unknown; '
        'do not accept full-data processing just because more images are now attached.'),
        lineage=bundle.get('lineage'), preview=metadata(bundle.get('preview', {})),
        accepted_pick_context=metadata(bundle.get('overlay', {})), error=bundle.get('error'),
        artifact=bundle.get('artifact'),
        requested_observations=metadata(bundle.get('tool_results', [])),
        current_pilot=metadata(bundle.get('current_pilot')))
    content = [{'type':'text', 'text':json.dumps(facts, default=str)}]
    for image in bundle.get('attachments', []):
        content.append({'type':'text', 'text':image.get('description') or image['role']})
        content.append({'type':'image_url', 'image_url':{'url':image['data_url']}})
    return {'role':'user', 'content':content}


async def collect_current_pilot(call, session, project, workspace, current, candidates,
                                facts_loader=lineage_facts):
    """Read paired current-lineage evidence, without changing candidates or model output."""
    picker = review_picker(candidates, current)
    if not picker:
        return None
    try:
        facts = await asyncio.to_thread(facts_loader, project, current, workspace, picker)
    except ValueError as exc:
        # Optional visual evidence must not fabricate missing jobs or abort the
        # controller. Transport exceptions still propagate to service recovery.
        return dict(success=False, attachments=[], error=dict(
            type=type(exc).__name__, message=str(exc)),
            note='Pilot lineage evidence is unavailable; do not infer missing observations.')
    overlay = await call(session, 'get_pick_inspection_visual_context', dict(
        project_uid=project, job_uid=facts['inspect_job_uid'], max_micrographs=2))
    uids = [str(p['micrograph_uid']) for p in overlay.get('panels', []) if 'micrograph_uid' in p][:2]
    if not uids:
        return dict(success=False, lineage=facts, overlay=overlay, attachments=[],
                    error='No accepted-pick micrograph IDs; do not infer missing particle boundaries')
    preview = await call(session, 'get_micrograph_preview_visual_context', dict(
        project_uid=project, job_uid=facts['inspect_job_uid'], output_name='micrographs',
        max_micrographs=2, micrograph_uids=uids))
    attachments = (full_images(preview, 'full_micrograph_with_scale') +
                   full_images(overlay, 'full_micrograph_pick_overlay'))
    scatter = overlay.get('score_scatter') or {}
    if scatter.get('data_url'):
        attachments.append(dict(role='accepted_pick_ncc_power',
            description='NCC/Power distribution after Inspect Picks filtering, not the original unfiltered population.',
            data_url=scatter['data_url']))
    return dict(success=bool(full_images(preview, 'full_micrograph_with_scale') and
                             full_images(overlay, 'full_micrograph_pick_overlay')),
                lineage=facts, overlay=overlay, preview=preview, attachments=attachments)


async def template_picker_visual_context(call, session, project, candidates, cryosift=False,
                                        threshold=3.0, timeout_seconds=300):
    recovery = next((c['pilot_template_recovery'] for c in candidates.get('candidate_actions', [])
                     if c.get('pilot_template_recovery')), None)
    if not recovery or not recovery.get('source_class_job'):
        return None
    uid = recovery['source_class_job']
    context = await call(session, 'get_class_average_visual_context', dict(
        project_uid=project, job_uid=uid, max_classes=0))
    context = dict(context, kind='pilot_template_picker',
                   selected_templates=recovery.get('selected_templates'),
                   source_select_job=recovery['source_select_job'])
    if cryosift:
        context['cryosift'] = await call(session, 'evaluate_2d_classes_with_cryosift', dict(
            project_uid=project, job_uid=uid, threshold=threshold, timeout_seconds=timeout_seconds))
    return context


def install(namespace, memory_root, workspace_uid, facts_loader=lineage_facts):
    """Wrap acquisition/prompt assembly, never model output or action validation."""
    from role_memory import write_json
    root = Path(memory_root)
    original_call = namespace['call_tool_json']
    original_prompt = namespace['build_autonomous_prompt']
    prompt_signature = inspect.signature(original_prompt)
    reviews, bundles = {}, {}

    async def call_tool(session, name, arguments):
        result = await original_call(session, name, arguments)
        if name == 'get_candidate_actions':
            key = (arguments.get('project_uid'), arguments.get('current_node_id'))
            reviews[key] = review_picker(result, key[1])
        if name != 'get_class_average_visual_context':
            return result
        key = (arguments.get('project_uid'), arguments.get('job_uid'))
        picker = reviews.get(key)
        if not picker:
            return result
        artifact = root / f'{key[0]}_{key[1]}_upstream.json'
        bundle = dict(version='pilot-current-upstream-v1', artifact=str(artifact), attachments=[])
        try:
            facts = await asyncio.to_thread(facts_loader, key[0], key[1], workspace_uid, picker)
            bundle['lineage'] = facts
            overlay = await original_call(session, 'get_pick_inspection_visual_context', dict(
                project_uid=key[0], job_uid=facts['inspect_job_uid'], max_micrographs=2))
            bundle['overlay'] = overlay
            uids = [str(p['micrograph_uid']) for p in overlay.get('panels', []) if 'micrograph_uid' in p][:2]
            if not uids:
                raise ValueError('Accepted-pick overlay has no calibrated micrograph IDs')
            preview = await original_call(session, 'get_micrograph_preview_visual_context', dict(
                project_uid=key[0], job_uid=facts['inspect_job_uid'], output_name='micrographs',
                max_micrographs=2, micrograph_uids=uids))
            bundle['preview'] = preview
            if preview.get('mcp_error') or preview.get('success') is False:
                raise ValueError('Calibrated preview tool failed: ' + str(preview.get('content'))[:1500])
            bundle['attachments'] = (full_images(preview, 'full_micrograph_with_scale') +
                                     full_images(overlay, 'full_micrograph_pick_overlay'))
            scatter = overlay.get('score_scatter') or {}
            if scatter.get('data_url'):
                bundle['attachments'].append(dict(role='accepted_pick_ncc_power',
                    description='NCC versus Power scores for the accepted Inspect Picks output, not the original unfiltered picker population.',
                    data_url=scatter['data_url']))
            bundle['success'] = (len(full_images(preview, 'full_micrograph_with_scale')) > 0 and
                                 len(full_images(overlay, 'full_micrograph_pick_overlay')) > 0)
            if not bundle['success']:
                raise ValueError('One or both paired full-field image types are unavailable')
        except Exception as exc:
            bundle.update(success=False, error=dict(type=type(exc).__name__, message=str(exc)))
        write_json(artifact, bundle)
        bundles[key] = bundle
        return result

    def prompt(*args, **kwargs):
        bound = prompt_signature.bind(*args, **kwargs)
        visual = bound.arguments.get('visual_context') or {}
        source = visual.get('source') or {}
        key = (source.get('project_uid'), source.get('job_uid'))
        messages = original_prompt(*args, **kwargs)
        if key in bundles and review_picker(bound.arguments['candidate_context']):
            messages.append(message_for(bundles[key]))
        return messages

    namespace['call_tool_json'], namespace['build_autonomous_prompt'] = call_tool, prompt
