"""Bound uncertain diameter decisions using restartable, lineage-local pilots."""
from __future__ import annotations

import copy
import os
import math

PILOT_PREFIX = 'diameter_trial_'
MICROGRAPHS = 20
MAX_ATTEMPTS = 3


def attempt_limit():
    """Allow a bounded, per-controller recovery extension; keep the default intact."""
    value = int(os.getenv('CRYOAGENT_DIAMETER_TRIAL_MAX_ATTEMPTS', str(MAX_ATTEMPTS)))
    if not 1 <= value <= 6:
        raise ValueError('CRYOAGENT_DIAMETER_TRIAL_MAX_ATTEMPTS must be between 1 and 6')
    return value


GUIDANCE = (
    'This is a small-sample particle-diameter trial, not a validated measurement. '
    'At Inspect Picks review, check complete particle boundaries and whether picks '
    'follow signal or background; high NCC/Power or many picks alone do not validate '
    'diameter. You may retry Blob Picker with a revised diameter before extraction. '
    'Otherwise continue the bounded extraction and Class 2D. After Class 2D, judge '
    'intact boundaries, coherent signal, centering, background, and diversity, using '
    'CryoSift as supporting evidence (lower scores are better). Do not assume that '
    'low class resolution numbers prove good picking or no truncation. Compare all '
    'completed pilot routes and promote the relatively best executable route, even '
    'when none meets an absolute quality target. Otherwise retry a revised '
    'diameter on the same small subset. Do not change a diameter when accepting full '
    'data: a revised diameter needs its own trial. Pilot Select 2D is allowed ONLY '
    'to obtain relatively better 2D templates for Template Picker on the full original '
    'CTF micrographs; never send pilot particles directly to Ab-initio or refinement. '
    'Blob Picker diameter is the MINIMUM search diameter, not a validated physical '
    'particle diameter. Do not blindly copy it into Template Picker diameter. '
    'For Template Picker estimate the whole particle extent from the selected '
    'class images and calibrated micrographs, and explain uncertainty. An exploratory '
    'poor-quality template can match background: inspect pick density, overlapping '
    'centers and contamination before trusting full-data counts. '
    'Use class images and CryoSift together: lower scores are better, and a successful '
    'empty selected_class_ids recommendation only means no class met its strict '
    'threshold, not that every class is unusable. Visually justified above-threshold '
    'selections require an explanation. Select relatively complete, coherent, centered views '
    'without a fixed retention quota; do not select junk just to continue. Prefer '
    'relatively better selected templates when available. Poor absolute quality or '
    'an empty CryoSift recommendation must not cause stop/request_input. '
    'New Template Picker particles must pass Inspect Picks, Extract, Class 2D and '
    'a fresh Select 2D before reconstruction. Normally retain the dimensionless '
    'Template Picker min_distance default. At the pilot attempt limit, choose among '
    'the available full-data promotion routes instead of stopping or requesting a '
    'person. Report quality risk honestly; promotion is exploratory, not proof of '
    'correct reconstruction. Preserve input validity, nonempty usable data, resource '
    'and Job budgets, API failure handling, and explicit user stop/pause. '
)


def full_promotion_options(state, current, source, full_count):
    from action_registry import find_ancestor_node_by_types, build_synthetic_blob_picker_candidate, _parameter
    options=[]
    for node in state.get('nodes',[]):
        if node.get('job_type')!='class_2D_new' or node.get('status')!='completed':
            continue
        if int(node['cryosparc_job_uid'][1:]) > int(current['cryosparc_job_uid'][1:]):
            continue
        if (node.get('outputs',{}).get('particles') or {}).get('num_items') == 0:
            continue
        picker=find_ancestor_node_by_types(state,node,{'blob_picker_gpu','template_picker_gpu'})
        if not picker or not is_trial_picker(picker):
            continue
        if not any(s.get('source_job_uid')==source['source_job_uid']
                   for s in picker.get('inputs',{}).get('micrographs',[])):
            continue
        parameters={key:_parameter(picker,key) for key in ('diameter','diameter_max')
                    if _parameter(picker,key) is not None}
        if not isinstance(parameters.get('diameter'),(int,float)) or parameters['diameter']<=0:
            continue
        action=build_synthetic_blob_picker_candidate(current,source)
        action['action_id']='diameter_full_'+node['cryosparc_job_uid']
        action['diameter_trial']=dict(role='accept_full',review=GUIDANCE,
            picker_uid=picker['cryosparc_job_uid'],class_job_uid=node['cryosparc_job_uid'],
            quality_status='exploratory_relative_best_not_validated')
        extraction=find_ancestor_node_by_types(state,node,{'extract_micrographs_multi'})
        inspection=find_ancestor_node_by_types(state,node,{'inspect_picks_v2'})
        action['diameter_trial']['tested_route']=dict(
            diameter=parameters.get('diameter'),diameter_max=parameters.get('diameter_max'),
            box_size_pix=_parameter(extraction,'box_size_pix') if extraction else None,
            inspect_job=inspection.get('cryosparc_job_uid') if inspection else None,
            inspect_parameters={key:_parameter(inspection,key) for key in
                ('ncc_score_thresh','lpower_thresh_min','lpower_thresh_max','calibrate_ncc','calibrate_pow')} if inspection else {})
        action['description']='Promote this tested route to ALL original CTF micrographs. '+GUIDANCE
        action['default_parameters']={**parameters,'num_process':full_count}
        for key,value in action['default_parameters'].items():
            action['parameter_template'].setdefault(key,{}).update(const=value,default=value)
        options.append(action)
    return sorted(options,key=lambda a:(a['diameter_trial']['class_job_uid']!=current['cryosparc_job_uid'],
                                        -int(a['diameter_trial']['class_job_uid'][1:])))


def guard_pilot_continuation(decision, candidates, visual_context):
    """Replace a quality-only terminal decision with an executable bounded promotion."""
    if (decision.get('decision_type') or decision.get('decision')) not in {'stop','complete','request_input'}:
        return decision,None
    choices=[c for c in candidates.get('candidate_actions',[]) if c.get('available',True)
             and (c.get('diameter_trial') or {}).get('role')=='accept_full']
    if not choices:
        return decision,None
    contexts=(visual_context or {}).get('contexts') or [visual_context or {}]
    def rank(candidate):
        uid=candidate['diameter_trial']['class_job_uid']
        score=math.inf
        for context in contexts:
            evidence=context.get('cryosift') or {}
            if not evidence.get('success') or evidence.get('job_uid')!=uid:
                continue
            if candidates.get('project_uid') and evidence.get('project_uid')!=candidates['project_uid']:
                continue
            weights={s['class_id']:s.get('particle_count',0) for s in context.get('class_statistics',[])}
            values=[(v['score'],weights.get(v['class_id'],0)) for v in evidence.get('class_scores',[])
                    if isinstance(v.get('score'),(int,float)) and math.isfinite(v['score'])]
            if values:
                total=sum(max(0,w) for _,w in values)
                score=sum(v*max(0,w) for v,w in values)/total if total else sum(v for v,_ in values)/len(values)
        return score,-int(uid[1:])
    chosen=min(choices,key=rank)
    score=rank(chosen)[0]
    method='particle-weighted CryoSift mean where counts exist, otherwise class mean; lower is better' if math.isfinite(score) else 'newest executable completed pilot; quantitative quality unavailable'
    result=dict(schema_version='2.0',decision_type='forward',action=chosen['job_type'],
        job_type=chosen['job_type'],action_id=chosen['action_id'],
        parameters=copy.deepcopy(chosen['default_parameters']),confidence=0.5,
        reason='User-authorized best-effort pilot promotion: continue the relatively best executable route instead of stopping for low absolute quality. '+method+'. This does not establish particle quality or reconstruction correctness.',
        risk_flags=['pilot_quality_unvalidated','automatic_relative_promotion'])
    return result,dict(original_decision=decision,executed_decision=result,ranking_method=method,
                       source_class_job=chosen['diameter_trial']['class_job_uid'],
                       score=score if math.isfinite(score) else None)


def is_trial_picker(node):
    return (node.get('job_type') == 'blob_picker_gpu'
            and str(node.get('title', '')).startswith('Agent ' + PILOT_PREFIX))


def initial_trial(candidate, full_count):
    trial = copy.deepcopy(candidate)
    count = min(MICROGRAPHS, full_count)
    trial['action_id'] = PILOT_PREFIX + candidate['action_id']
    trial['diameter_trial'] = {'role': 'pilot', 'micrographs': count}
    trial.setdefault('default_parameters', {})['num_process'] = count
    trial.setdefault('parameter_template', {})['num_process'] = {
        'type': 'integer', 'minimum': 1, 'const': count, 'default': count}
    trial['description'] = (
        f'If diameter confidence is uncertain, test the chosen diameter on only {count} '
        'micrographs, then Inspect Picks, Extract and Class 2D before full-data picking.')
    return trial


def trial_candidates(state, current, candidates):
    """Return None for normal branches; no saved runner state is required to resume."""
    from action_registry import (
        find_ancestor_node_by_types, output_source, output_item_count,
        build_synthetic_blob_picker_candidate, build_synthetic_template_picker_candidate,
        template_source_from_node, action_references_node, _parameter,
    )
    picker = (current if current.get('job_type') in {'blob_picker_gpu', 'template_picker_gpu'}
              else find_ancestor_node_by_types(state, current, {'blob_picker_gpu', 'template_picker_gpu'}))
    if not picker or not is_trial_picker(picker):
        return None
    # Only the explicit pilot funnel is handled here; full-data branches have a
    # new non-pilot picker and therefore cannot be mistaken for calibration jobs.
    stage = current.get('job_type')
    if stage not in {'blob_picker_gpu', 'inspect_picks_v2',
                     'extract_micrographs_multi', 'class_2D_new', 'select_2D'}:
        return None
    ctf = find_ancestor_node_by_types(state, picker, {
        'patch_ctf_estimation_multi', 'patch_ctf_estimation', 'patch_ctf',
        'ctf_estimation', 'ctf_estimation_gctf'})
    source = output_source(ctf, 'exposures') if ctf else None
    full_count = output_item_count(ctf, 'exposures') if ctf else None
    if not source or not full_count:
        return []  # Fail closed instead of leaking a pilot into reconstruction.
    if stage == 'select_2D':
        templates = template_source_from_node(current)
        if not templates or templates['source_output'] != 'templates_selected':
            return []
        count = (current.get('outputs', {}).get('templates_selected') or {}).get('num_items')
        if count is not None and int(count) <= 0:
            return []
        action = build_synthetic_template_picker_candidate(current, source, templates)
        action['description'] = GUIDANCE
        action['pilot_template_recovery'] = dict(source_select_job=current['cryosparc_job_uid'],
            pilot_picker=picker['cryosparc_job_uid'], full_ctf_job=ctf['cryosparc_job_uid'],
            source_class_job=(find_ancestor_node_by_types(state, current, {'class_2D_new'}) or {}).get('cryosparc_job_uid'),
            selected_templates=_parameter(current, 'selected_templates'))
        return [action,*full_promotion_options(state,current,source,full_count)]
    sample_count = min(MICROGRAPHS, full_count)
    source_uid = source['source_job_uid']
    attempts = sum(1 for node in state.get('nodes', []) if is_trial_picker(node)
                   and any(s.get('source_job_uid') == source_uid
                           for s in (node.get('inputs', {}).get('micrographs') or [])))
    base = build_synthetic_blob_picker_candidate(current, source)
    base['default_parameters'] = {
        key: _parameter(picker, key) for key in ('diameter', 'diameter_max')
        if _parameter(picker, key) is not None}
    for key, value in base['default_parameters'].items():
        base['parameter_template'].setdefault(key, {})['default'] = value
    retry = initial_trial(base, full_count)
    retry['diameter_trial'].update(role='retry', attempt=attempts + 1,
                                   review=GUIDANCE, picker_uid=picker['cryosparc_job_uid'])
    result = []
    if stage == 'class_2D_new':
        result.extend(full_promotion_options(state,current,source,full_count))
        for candidate in candidates:
            if (candidate.get('job_type') == 'select_2D' and candidate.get('available')
                    and action_references_node(candidate, current['cryosparc_job_uid'])):
                selection = copy.deepcopy(candidate)
                selection['diameter_trial'] = dict(role='select_templates', review=GUIDANCE,
                    picker_uid=picker['cryosparc_job_uid'])
                selection['description'] = GUIDANCE
                result.append(selection)
    else:
        next_type = {'blob_picker_gpu': 'inspect_picks_v2',
                     'inspect_picks_v2': 'extract_micrographs_multi',
                     'extract_micrographs_multi': 'class_2D_new'}[stage]
        # Exclude observed historical children connected to other branches.
        for action in candidates:
            if action.get('job_type') != next_type:
                continue
            sources = [s for group in (action.get('required_inputs') or {}).values() for s in group]
            if not any(s.get('source_job_uid') == current['cryosparc_job_uid'] for s in sources):
                continue
            action = copy.deepcopy(action)
            action['diameter_trial'] = {'role': 'continue', 'review': GUIDANCE,
                                        'picker_uid': picker['cryosparc_job_uid']}
            if next_type == 'extract_micrographs_multi':
                action.setdefault('default_parameters', {})['num_extract'] = sample_count
                action.setdefault('parameter_template', {})['num_extract'] = {
                    'type': 'integer', 'minimum': 1, 'const': sample_count, 'default': sample_count}
            result.append(action)
    if stage in {'inspect_picks_v2', 'class_2D_new'} and attempts < attempt_limit():
        retry['description'] = 'Revise particle diameter and retry the same small subset. ' + GUIDANCE
        result.append(retry)
    return result


def enabled():
    return os.getenv('CRYOAGENT_UNCERTAIN_DIAMETER_TRIAL', '1').lower() not in {'0', 'false', 'off'}
