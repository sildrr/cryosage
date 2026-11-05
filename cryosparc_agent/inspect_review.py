"""Bounded Inspect Picks feedback using actual accepted/rejected particle IDs."""
import math
import os

THRESHOLDS = ('ncc_score_thresh', 'lpower_thresh_min', 'lpower_thresh_max')
INSTRUCTIONS = (
    'Review the COMPLETED Inspect Picks result before extraction. The same full-field '
    'micrographs are supplied unmarked and with accepted candidates in green and '
    'rejected candidates in magenta. Colors encode filter membership, NOT quality, '
    'and marker radii are not molecular boundaries. '
    'Look for plausible real particles that were rejected, retained background/junk, '
    'overlapping duplicate centers, and uncertainty from compression. A real particle '
    'with no original picker candidate cannot be recovered merely by loosening Inspect. '
    'Use the original NCC/Power distribution and applied raw thresholds to locate the '
    'problem; either lower or raise bounds as evidence supports. No preferred count or '
    'retention fraction exists. Choose an offered inspect_review action to revise '
    'thresholds on the ORIGINAL picker candidates, never on the already filtered subset. '
    'Change at least one threshold, keep calibrate_ncc=false, calibrate_pow=false and '
    'do_auto_cluster=false, and include a finite Power upper bound greater than its '
    'lower bound. Otherwise continue the offered extraction or existing pilot retry. '
    'At most two threshold revisions per picker are allowed; do not request human input '
    'or repeat the same thresholds merely because visual evidence is uncertain. At the '
    'limit continue the available downstream route with an honest uncertainty statement. '
    'Explain specific retained/rejected examples and expected effects, without inventing '
    'particle identities or claiming these images prove reconstruction quality.'
)


def particle_source(node):
    sources = (node.get('inputs') or {}).get('particles') or []
    if len(sources) == 1:
        return sources[0].get('source_job_uid'), sources[0].get('source_output')
    return None


def review_candidate(state, current, candidates):
    if not current or current.get('job_type') != 'inspect_picks_v2' or current.get('status') != 'completed':
        return None
    if not any(c.get('available', True) and c.get('job_type') == 'extract_micrographs_multi' for c in candidates):
        return None
    params = current.get('key_parameters') or {}
    if params.get('calibrate_ncc') or params.get('calibrate_pow'):
        return None
    limit = int(os.getenv('CRYOAGENT_INSPECT_REVIEW_MAX_REVISIONS', '2'))
    if not 0 <= limit <= 3:
        raise ValueError('Inspect review revision limit must be 0..3')
    source = particle_source(current)
    if not source or not all(source):
        return None
    siblings = [n for n in state.get('nodes', [])
                if n.get('job_type') == 'inspect_picks_v2' and particle_source(n) == source]
    # Native Jobs provide durable accounting across controller restarts. Pending,
    # failed and killed revisions also consume this bounded attempt allowance.
    if len(siblings) >= 1 + limit or any(n.get('status') not in {'completed', 'failed', 'killed'} for n in siblings):
        return None
    from action_registry import build_candidate_action
    candidate = build_candidate_action(state, current, 'forward')
    candidate.update(action_id='inspect_review_' + current['cryosparc_job_uid'],
        execution_mode='create_job', reference_job_uid=None, reference_status=None,
        description=INSTRUCTIONS,
        inspect_review=dict(source_inspect_job=current['cryosparc_job_uid'],
                            revision=len(siblings), maximum_revisions=limit))
    candidate['default_parameters'] = {k: v for k, v in params.items() if k in THRESHOLDS}
    for k in ('calibrate_ncc', 'calibrate_pow', 'do_auto_cluster'):
        candidate['default_parameters'][k] = False
        candidate['parameter_template'][k] = dict(type='boolean', const=False, default=False)
    for k in THRESHOLDS:
        candidate['parameter_template'].setdefault(k, dict(type='number'))['required'] = True
    return candidate if candidate.get('available') else None


def threshold_error(parameters, candidate):
    values = {**candidate['default_parameters'], **parameters}
    for key in THRESHOLDS:
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return 'Inspect review requires finite raw NCC and Power bounds'
    if values['lpower_thresh_max'] <= values['lpower_thresh_min']:
        return 'Power upper bound must exceed lower bound'
    if any(values.get(k) is not False for k in ('calibrate_ncc', 'calibrate_pow', 'do_auto_cluster')):
        return 'Inspect review must use raw scores without auto clustering'
    if all(values[k] == candidate['default_parameters'].get(k) for k in THRESHOLDS):
        return 'Inspect review must change at least one threshold; otherwise continue extraction'
    return None


def membership(raw_ids, accepted_ids):
    import numpy as np
    raw = np.asarray(raw_ids, dtype=np.uint64)
    accepted = np.asarray(accepted_ids, dtype=np.uint64)
    if len(np.unique(raw)) != len(raw) or len(np.unique(accepted)) != len(accepted):
        raise ValueError('Particle IDs must be unique for matched review')
    mask = np.isin(raw, accepted)
    if int(mask.sum()) != len(accepted):
        raise ValueError('Accepted IDs are not a subset of original picker candidates')
    return mask


def score_statistics(ncc, power, mask):
    import numpy as np
    result = {}
    for label, select in (('original', np.ones(len(mask), dtype=bool)), ('accepted', mask), ('rejected', ~mask)):
        result[label] = {}
        for name, values in (('ncc_score', ncc), ('power', power)):
            data = np.asarray(values)[select]
            data = data[np.isfinite(data)]
            result[label][name] = (dict(zip(('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max'),
                map(float, np.percentile(data, [0, 5, 25, 50, 75, 95, 100])))) if len(data) else {})
    return result


def build_visual_context(project_uid, job_uid, max_micrographs=3):
    import numpy as np
    from PIL import ImageDraw
    from cryosparc_client import cryosparc_client
    from vision_inputs import (load_micrograph_image, render_full_micrograph_preview,
        micrograph_pixel_size, full_field_pick_position, redraw_full_field_scale_bar,
        _save_visual_evidence, vision_cache_dir, build_pick_score_scatter)
    if not 1 <= max_micrographs <= 3:
        raise ValueError('Inspect review supports 1..3 matched micrographs')
    project = cryosparc_client().find_project(project_uid)
    job = project.find_job(job_uid)
    if job.type != 'inspect_picks_v2' or job.status != 'completed':
        raise ValueError('A completed Inspect Picks job is required')
    spec = job.model.model_dump()['spec']
    connections = spec['inputs']['particles']['connections']
    if len(connections) != 1:
        raise ValueError('A single original candidate source is required')
    parent = project.find_job(connections[0]['job_uid'])
    raw = parent.load_output(connections[0]['output'])
    accepted = job.load_output('particles')
    mask = membership(raw['uid'], accepted['uid'])
    micrographs = job.load_output('micrographs')
    mics = np.asarray(raw['location/micrograph_uid'], dtype=np.uint64)
    xs, ys = raw['location/center_x_frac'], raw['location/center_y_frac']
    ncc = np.asarray(raw['pick_stats/ncc_score'])
    power = np.asarray(raw['pick_stats/power'])
    uids, counts = np.unique(mics, return_counts=True)
    # Matched fields span sparse, middle and dense raw populations, not separate
    # before/after samples which would confound the visual comparison.
    order = np.argsort(counts)
    positions = np.linspace(0, len(order) - 1, min(max_micrographs, len(order))).round().astype(int)
    mic_index = {int(v): i for i, v in enumerate(micrographs['uid'])}
    cache = vision_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    evidence, panels, examples = [], [], []
    for number, position in enumerate(positions, 1):
        uid = int(uids[order[position]])
        idx = np.flatnonzero(mics == uid)
        mi = mic_index[uid]
        image, calibration = render_full_micrograph_preview(
            load_micrograph_image(project, str(micrographs['micrograph_blob/path'][mi]), None),
            micrograph_pixel_size(micrographs, mi))
        clean = image.copy()
        draw = ImageDraw.Draw(image)
        shown = {}
        for kept, color in ((False, (255, 70, 220)), (True, (40, 255, 80))):
            group = idx[mask[idx] == kept]
            chosen = group[np.linspace(0, len(group) - 1, min(600, len(group))).astype(int)] if len(group) else group
            shown['accepted' if kept else 'rejected'] = len(chosen)
            for p in chosen:
                x, y = full_field_pick_position(xs[p], ys[p], calibration)
                draw.ellipse((x-4, y-4, x+4, y+4), outline=color, width=2)
            marked = chosen[np.linspace(0, len(chosen)-1, min(2, len(chosen))).astype(int)] if len(chosen) else chosen
            for label_index, p in enumerate(marked, 1):
                label = f"M{number}{'A' if kept else 'R'}{label_index}"
                x, y = full_field_pick_position(xs[p], ys[p], calibration)
                draw.text((min(960, max(4, x+6)), min(940, max(28, y))), label,
                          fill='white', stroke_width=2, stroke_fill='black')
                examples.append(dict(label=label, micrograph_uid=uid, accepted=kept,
                    ncc_score=float(ncc[p]), power=float(power[p]),
                    display_center=[float(x), float(y)]))
        draw.rectangle((0, 0, 1024, 25), fill='black')
        draw.text((5, 6), 'GREEN=accepted MAGENTA=rejected (membership, NOT quality)', fill='white')
        redraw_full_field_scale_bar(image, calibration)
        for role, picture in (('inspect_review_unmarked', clean), ('inspect_review_membership', image)):
            evidence.append(_save_visual_evidence(picture, cache,
                f'{project_uid}_{job_uid}_{role}_{number}.png', role,
                f'Matched full micrograph uid={uid}; green accepted, magenta rejected; sampled overlays, not quality labels.'))
        panels.append(dict(micrograph_uid=uid, input_count=len(idx), accepted_count=int(mask[idx].sum()),
                           rejected_count=int((~mask[idx]).sum()), shown=shown, calibration=calibration))
    scatter = _save_visual_evidence(build_pick_score_scatter(ncc, power), cache,
        f'{project_uid}_{job_uid}_review_raw_density.png', 'inspect_review_raw_density',
        'Original unfiltered raw NCC/Power density; compare numeric thresholds, brightness is count density.')
    return dict(kind='inspect_result_review', source=dict(project_uid=project_uid, job_uid=job_uid),
        picker_job_uid=parent.uid, applied_parameters=spec['params'], input_count=len(mask),
        accepted_count=int(mask.sum()), rejected_count=int((~mask).sum()),
        retained_fraction=float(mask.mean()) if len(mask) else 0.0, panels=panels,
        evidence_images=evidence, score_scatter=scatter,
        image_count=len(evidence) + 1, score_statistics=score_statistics(ncc, power, mask),
        annotated_examples=examples,
        review_instruction=INSTRUCTIONS, score_space='raw_pick_stats')
