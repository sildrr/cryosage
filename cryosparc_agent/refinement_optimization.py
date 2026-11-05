"""Bounded, source-scoped post-refinement interventions and continuations."""
from __future__ import annotations

from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re

import numpy as np

REFINES = {'homo_refine_new', 'nonuniform_refine_new', 'refine_3D_new'}
PARTICLE_OPERATIONS = {'agent_reference_union', 'agent_scale_cleanup'}
OPTICS = dict(refine_defocus_refine=True, refine_ctf_global_refine=True,
    crl_df_range=1000., crl_min_res_A=20., crg_min_res_A=10.,
    crg_do_tilt=True, crg_do_trefoil=True, crg_do_spherical=True,
    crg_do_tetrafoil=False, crg_do_anisomag=False)
GUIDANCE = (
    'Validated post-refinement options are available as explicit candidates. Prefer one '
    'modern NU comparison after a plausible Homo baseline even when the target gap exceeds '
    '0.1 A. At a coherent near-atomic baseline (about 4 A or better), prefer the one-time '
    'optics_refinement candidate to fit per-particle defocus plus global beam tilt, trefoil '
    'and Cs before repeatedly changing picking or selection. This is a changed-parameter '
    'experiment, not an unchanged-refine loop. Reference-assisted class expansion uses a '
    '15 A lowpassed own volume and unions its selected classes with the original selection; '
    'it does not guarantee angular coverage, so inspect class/projection and orientation '
    'evidence. Low-scale cleanup is optional, one mild lowest-5% cut per lineage, recalculated '
    'from that result, not a transferable numeric threshold or proof of bad particles. '
    'Do not recursively trim the lowest 5%. Preserve all best checkpoints. A 0.02 A gain '
    'in one run is not decisive evidence. Complete a selected intervention through its '
    'fixed downstream NU before planning another experiment. Ordinary reselect_2d remains '
    'a strict subset; only the explicit reference_class_expansion route can add classes. '
    'For imported-particle datasets these post-refinement options remain applicable: '
    'use the imported per-particle CTF metadata, the current Class 2D population and '
    'your own reconstructed volume. Do not require micrographs, picking or extraction '
    'ancestors for reference-assisted class selection, symmetry comparisons or optics '
    'refinement. A symmetry recommendation is a hypothesis: validate it against the '
    'preserved same-input baseline before applying it to subsequent experiments. '
)


def enabled():
    return os.getenv('CRYOAGENT_REFINEMENT_OPTIMIZATION', '1').lower() in {'1', 'true', 'yes', 'on'}


def ancestors(state, node):
    lookup = {n['cryosparc_job_uid']: n for n in state.get('nodes', [])}
    result, todo, seen = [], [node], set()
    while todo:
        current = todo.pop(0)
        uid = current['cryosparc_job_uid']
        if uid in seen:
            continue
        seen.add(uid)
        result.append(current)
        todo.extend(lookup[p] for p in current.get('parent_job_uids', []) if p in lookup)
    return result


def resolution(node):
    stats = ((node.get('outputs', {}).get('volume') or {}).get('latest_summary_stat_values') or {})
    value = (stats.get('fsc_info') or {}).get('radwn_final_A')
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else None


def nu_parameters(reference, optics=False):
    old = reference.get('key_parameters') or {}
    params = dict(refine_symmetry=old.get('refine_symmetry') or 'C1', refine_res_init=30.,
        refine_gs_resplit=False, refine_nu_enable=True, refine_do_marg=True,
        refine_num_final_iterations=2,
        refine_defocus_refine=bool(old.get('refine_defocus_refine')),
        refine_ctf_global_refine=bool(old.get('refine_ctf_global_refine')))
    for key in OPTICS:
        if key in old:
            params[key] = old[key]
    if optics:
        params.update(OPTICS)
    return params


def candidate(identifier, kind, sources, params, description, *, rollback_kind=None):
    from job_specs import get_parameter_template
    template = get_parameter_template(kind)
    for key, value in params.items():
        value_type = 'boolean' if isinstance(value, bool) else 'integer' if isinstance(value, int) else 'number' if isinstance(value, float) else 'string'
        template[key] = {**template.get(key, {}), 'type': value_type, 'default': value, 'const': value}
    result = dict(action_id=identifier, action_type='forward', workflow_node_id=identifier,
        reference_job_uid=next(iter(sources.values()))[0], reference_status='completed',
        job_type=kind, execution_mode='create_job', available=True, blocked_by=[],
        required_inputs={key: [dict(source_job_uid=uid, source_output=output)]
                         for key, (uid, output) in sources.items()},
        missing_required_inputs=[], parameter_template=template, default_parameters=params,
        description=description, optimization_contract=True)
    if rollback_kind:
        result.update(refinement_rollback=True, quality_rollback=True, rollback_kind=rollback_kind)
    else:
        result['optimization_continuation'] = True
    return result


@lru_cache(maxsize=128)
def reference_sampling(project_uid, class_uid, reference_uid):
    from cryosparc_client import cryosparc_client
    project = cryosparc_client().find_project(project_uid)
    volume = project.find_job(reference_uid).load_output('volume')
    classes = project.find_job(class_uid).load_output('class_averages')
    shape = np.asarray(classes['blob/shape'])
    physical = np.asarray(volume['map/shape'][0]) * float(volume['map/psize_A'][0])
    if not len(shape) or not np.all(shape == shape[0]) or shape[0, 0] != shape[0, 1]:
        raise ValueError('Reference selection requires one shared square Class2D image size')
    if not np.allclose(shape * classes['blob/psize_A'][:, None], physical[0], atol=.02, rtol=1e-4):
        raise ValueError('Reference/Class2D physical field of view mismatch')
    if not np.allclose(physical, physical[0]):
        raise ValueError('Reference volume is not cubic in physical units')
    return int(shape[0, 0])


def build_candidates(state, current, project_uid):
    if not enabled() or current.get('job_type') not in REFINES or current.get('status') != 'completed':
        return []
    valid = [n for n in state.get('nodes', []) if n.get('status') == 'completed'
             and n.get('job_type') in REFINES and resolution(n) is not None
             and all((n.get('outputs', {}).get(k) or {}).get('available') for k in ('particles', 'volume'))]
    if not valid:
        return []
    reference = min(valid, key=resolution)
    uid = reference['cryosparc_job_uid']
    lineage = ancestors(state, reference)
    titles = [n.get('title') or '' for n in state.get('nodes', [])]
    actions = []
    old = reference.get('key_parameters') or {}
    if resolution(reference) <= 4. and not (old.get('refine_defocus_refine') and old.get('refine_ctf_global_refine')):
        identifier = 'opt_optics_' + uid
        if not any(identifier in title for title in titles):
            action = candidate(identifier, 'nonuniform_refine_new',
                dict(particles=(uid, 'particles'), volume=(uid, 'volume')),
                nu_parameters(reference, optics=True),
                'One same-particle C1/known-symmetry NU with defocus and global tilt/trefoil/Cs fitting; preserve half sets and best result.',
                rollback_kind='optics_refinement')
            action['refinement_strategy_switch'] = True
            actions.append(action)
    if not any('opt_scale_' in (n.get('title') or '') for n in lineage):
        identifier = 'opt_scale_' + uid
        if not any(identifier in title for title in titles):
            actions.append(candidate(identifier, 'agent_scale_cleanup',
                dict(particles=(uid, 'particles'), volume=(uid, 'volume')), dict(quantile=.05),
                'Optional one-time lowest-5% optimal-scale exclusion, then NU from the same map. No recursive percentile pruning.',
                rollback_kind='low_scale_cleanup'))
    selection = next((n for n in lineage if n.get('job_type') == 'select_2D'), None)
    if not selection or not project_uid:
        return actions
    classes = next((n for n in ancestors(state, selection) if n.get('job_type') in {'class_2D_new', 'class_2D'}), None)
    if not classes:
        return actions
    class_uid, select_uid = classes['cryosparc_job_uid'], selection['cryosparc_job_uid']
    if any('opt_reference_lowpass_' in title and f'_{class_uid}_' in title for title in titles):
        return actions
    try:
        size = reference_sampling(project_uid, class_uid, uid)
    except ValueError:
        # A different-box branch can be the workspace best. Never silently
        # compare mismatched fields of view or hide other valid interventions.
        return actions
    actions.append(candidate(f'opt_reference_lowpass_{uid}_{class_uid}_{select_uid}', 'volume_tools',
        dict(volume=(uid, 'volume')), dict(vol_lpf_A=15., vol_resample_N=size),
        'Use the best own volume at 15 A to find compatible additional classes. Preserve original selection by UID union; then NU, no new Ab-initio.',
        rollback_kind='reference_class_expansion'))
    return actions


def continuation(state, current):
    """Only tagged optimization jobs get a fixed continuation; no general DAG override."""
    if not enabled() or current.get('status') != 'completed':
        return None
    title, current_uid = current.get('title') or '', current['cryosparc_job_uid']
    match = re.search(r'\bopt_reference_(lowpass|match|union)_(J\d+)_(J\d+)_(J\d+)\b', title)
    if match:
        stage, ref, classes, selection = match.groups()
        lookup = {n['cryosparc_job_uid']: n for n in state['nodes']}
        if not all(uid in lookup and lookup[uid].get('status') == 'completed' for uid in (ref, classes, selection)):
            raise ValueError('Optimization continuation references unavailable workspace sources')
        suffix = f'{ref}_{classes}_{selection}'
        if stage == 'lowpass':
            return [candidate('opt_reference_match_' + suffix, 'reference_select_2D',
                dict(particles=(classes, 'particles'), templates=(classes, 'class_averages'), volume=(current_uid, 'volume')),
                dict(select_mode='cluster', res_filter=20., max_res=15., lp_filter=True),
                'Match all original Class2D images against the lowpassed own reference. Cluster in correlation-score space, not view space.')]
        if stage == 'match':
            return [candidate('opt_reference_union_' + suffix, 'agent_reference_union',
                dict(source=(classes, 'particles'), original=(selection, 'particles_selected'),
                     proposed=(current_uid, 'particles_selected'), volume=(ref, 'volume')), {},
                'Union reference-selected UIDs with the original selection; restore common original Class2D metadata and preserve source identity.')]
        return [candidate('opt_reference_refine_' + suffix, 'nonuniform_refine_new',
            dict(particles=(current_uid, 'selected'), volume=(ref, 'volume')),
            nu_parameters(lookup[ref]), 'Refine the expanded original-metadata population against the source volume; do not rerun Ab-initio.')]
    match = re.search(r'\bopt_scale_(J\d+)\b', title)
    if match:
        ref = match.group(1)
        reference = next((n for n in state['nodes'] if n['cryosparc_job_uid'] == ref), None)
        if reference is None or reference.get('status') != 'completed':
            raise ValueError('Scale cleanup reference unavailable')
        return [candidate('opt_scale_refine_' + ref, 'nonuniform_refine_new',
            dict(particles=(current_uid, 'selected'), volume=(ref, 'volume')),
            nu_parameters(reference), 'Refine the mildly scale-cleaned particles once; retain native poses, CTF and half-set assignments.')]
    return None


def select_population(kind, datasets, params):
    """Return an unchanged-metadata subset plus an auditable selection record."""
    if kind == 'agent_scale_cleanup':
        source = datasets['particles']
        values = np.asarray(source['alignments3D/alpha_min'], dtype=float)
        if params != {'quantile': .05} or not np.all(np.isfinite(values)):
            raise ValueError('Scale cleanup requires finite optimal scales and the validated p05 policy')
        threshold = float(np.quantile(values, .05))
        keep = np.flatnonzero(values > threshold)
        fraction = 1 - len(keep) / len(source)
        if len(keep) < 1000 or not .04 <= fraction <= .06:
            raise ValueError('Percentile ties or population size make p05 cleanup unsafe')
        return source.take(keep), dict(source_count=len(source), selected_count=len(keep),
            field='alignments3D/alpha_min', threshold=threshold, quantile=.05,
            excluded_fraction=fraction, ties_not_broken=True)
    if kind != 'agent_reference_union' or params:
        raise ValueError('Unknown particle operation or unsupported parameters')
    source = datasets['source']
    original, proposed = datasets['original']['uid'], datasets['proposed']['uid']
    combined = np.union1d(original, proposed)
    if len(np.unique(source['uid'])) != len(source) or not np.all(np.isin(combined, source['uid'])):
        raise ValueError('Reference selection contains duplicate or out-of-source identities')
    indices = np.flatnonzero(np.isin(source['uid'], combined))
    return source.take(indices), dict(source_count=len(source), original_count=len(original),
        proposed_count=len(proposed), selected_count=len(indices),
        added_count=int(len(np.setdiff1d(combined, original))), original_metadata_preserved=True)


def execute_particle_operation(cs, project_uid, workspace_uid, plan):
    """A single counted External job; no hidden native jobs or API model calls."""
    kind = plan['job_type']
    connections = plan.get('connections') or {}
    required = {'particles', 'volume'} if kind == 'agent_scale_cleanup' else {'source', 'original', 'proposed', 'volume'}
    if set(connections) != required:
        raise ValueError('Particle operation input contract mismatch')
    project = cs.find_project(project_uid)
    workspace = project.find_workspace(workspace_uid)
    members = {j.uid: j for j in workspace.find_jobs()}
    title = 'Agent ' + plan['action_id']
    previous = [j for j in members.values() if j.title == title]
    if len(previous) > 1 or (previous and previous[0].status != 'completed'):
        raise RuntimeError('Ambiguous/incomplete previous particle operation; reconcile before retry')
    if previous:
        return dict(success=True, dry_run=False, status='completed', queued=False,
            job_uid=previous[0].uid, job_type=kind, project_uid=project_uid,
            workspace_uid=workspace_uid, reused=True, planned_action=plan)
    datasets = {}
    for name, connection in connections.items():
        if not isinstance(connection, tuple) or len(connection) != 2:
            raise ValueError('Particle operation requires one exact source per input')
        uid, output = connection
        if uid not in members or members[uid].status != 'completed':
            raise ValueError('Particle operation source is not completed in this workspace')
        if name != 'volume':
            datasets[name] = members[uid].load_output(output)
    selected, audit = select_population(kind, datasets, plan.get('resolved_parameters') or {})
    slots = sorted({field.split('/')[0] for field in selected.fields() if '/' in field})
    source_name = 'particles' if kind == 'agent_scale_cleanup' else 'source'
    job = cs.create_external_job(project_uid, workspace_uid, title=title,
        desc='Bounded particle curation; ' + json.dumps(audit, sort_keys=True))
    try:
        for name, (uid, output) in connections.items():
            input_slots = ['map'] if name == 'volume' else sorted({
                field.split('/')[0] for field in datasets[name].fields() if '/' in field})
            job.connect(name, uid, output, slots=input_slots)
        job.add_output('particle', 'selected', slots=slots, passthrough=source_name)
        job.start()
        job.save_output('selected', selected)
        saved = job.load_output('selected')
        for field in selected.fields():
            a, b = selected[field], saved[field]
            same = np.array_equal(a, b, equal_nan=True) if np.issubdtype(a.dtype, np.inexact) else np.array_equal(a, b)
            if not same:
                raise ValueError('Saved particle metadata changed: ' + field)
        job.log(json.dumps(audit, sort_keys=True))
        job.stop()
    except Exception as exc:
        try:
            job.stop(error=str(exc))
        except Exception:
            pass
        raise
    return dict(success=True, dry_run=False, status='completed', queued=False, job_uid=job.uid,
        job_type=kind, project_uid=project_uid, workspace_uid=workspace_uid,
        audit=audit, planned_action=plan, human_action_required=False)
