"""Source-bound human exceptions for pilot shape diversity, never particle quality."""
import copy
import hashlib
import json
import math
import uuid
from pathlib import Path


def diversity_only(gate):
    count = gate.get('strict_core_particles', gate.get('qualified_particles'))
    minimum = max(2001, (gate.get('policy') or {}).get('min_particles', 2001))
    return (gate.get('available') is True and gate.get('passed') is False
            and set(gate.get('failed_conditions') or []) == {'insufficient_supported_2d_shape_groups'}
            and type(count) in (int, float) and math.isfinite(count) and count >= minimum)


def approved_diversity(gate):
    approval = gate.get('manual_diversity_approval') or {}
    return (diversity_only(gate) and approval.get('scope') == 'pilot_shape_diversity_only'
            and approval.get('approved') is True
            and all(isinstance(approval.get(k), str) and approval[k].strip()
                    for k in ('id', 'approved_at', 'reason', 'fingerprint')))


def fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def candidates(root):
    root = Path(root)
    if not (root/'manifest.json').exists() or not (root/'state.json').exists():
        return []
    manifest = json.loads((root/'manifest.json').read_text())
    state = json.loads((root/'state.json').read_text())
    if (manifest.get('production_pipeline') != 'web-pilot-to-full-v1'
            or state.get('pilot_accepted') or state.get('full')
            or state.get('status') != 'protocol_exhausted'):
        return []
    result = []
    for bucket in ('sequential_trials', 'diameter_recovery_trials'):
        for tag, record in state.get(bucket, {}).items():
            row, reconstruction = record.get('row') or {}, record.get('reconstruction') or {}
            gate = reconstruction.get('gate') or {}
            resolution = reconstruction.get('best_resolution_A')
            if (not row.get('executable') or not row.get('classes') or not diversity_only(gate)
                    or not reconstruction.get('selections') or not reconstruction.get('best_refine')
                    or type(resolution) not in (int, float) or not math.isfinite(resolution) or resolution <= 0):
                continue
            result.append(dict(bucket=bucket, tag=tag, fingerprint=fingerprint(record),
                class_job=row['classes'], best_refine=reconstruction['best_refine'],
                resolution_A=resolution, micrographs=row['count'],
                qualified_particles=gate.get('strict_core_particles', gate.get('qualified_particles')),
                supported_group_count=gate.get('supported_group_count')))
    return sorted(result, key=lambda x: (x['resolution_A'], -x['qualified_particles'], x['tag']))


def request_approval(root, reason, requested_by='authenticated_web_operator'):
    from webapp.core import now, write
    root = Path(root)
    reason = str(reason).strip()
    if not reason or len(reason) > 4000:
        raise ValueError('A human approval reason of 1-4000 characters is required')
    choices = candidates(root)
    if not choices:
        raise ValueError('No stopped pilot with sufficient good particles and ONLY a diversity failure')
    approval = dict(choices[0], id=uuid.uuid4().hex, approved=True, approved_at=now(),
        requested_by=requested_by, reason=reason, scope='pilot_shape_diversity_only',
        route_policy='best_existing_pilot_refine_then_same_blob_route_to_full',
        budget_policy='use_remaining_original_total_budget_without_increase')
    write(root/'pilot_diversity_approval.json', approval)
    write(root/'pilot_diversity_approvals'/f"{approval['id']}.json", approval)
    return approval


def apply_approval(experiment):
    path = experiment.root/'pilot_diversity_approval.json'
    if not path.exists() or experiment.state.get('pilot_accepted'):
        return None
    approval = json.loads(path.read_text())
    record = experiment.state[approval['bucket']][approval['tag']]
    if fingerprint(record) != approval['fingerprint']:
        raise ValueError('Approved pilot evidence changed; obtain a new human approval')
    row, outcome = record['row'], record['reconstruction']
    gate = copy.deepcopy(outcome['gate'])
    gate['manual_diversity_approval'] = approval
    if not approved_diversity(gate):
        raise ValueError('Approval cannot waive particle quality, missing evidence, or other failures')
    from two_tier_picker import scores_and_counts
    strict_count, seen, sources = 0, set(), []
    for selection in outcome['selections']:
        if selection['source']['project_uid'] != experiment.project.uid:
            raise ValueError('Approved class source belongs to another project')
        source_uid = selection['source']['job_uid']
        context = (json.loads((experiment.root/'evidence'/row['evidence_slug']/'context.json').read_text())
                   if source_uid == row['classes'] else experiment.classes('manual_approval_'+source_uid,
                       experiment.project.find_job(source_uid)))
        if context['source'] != selection['source']:
            raise ValueError('Approved class source mismatch')
        scores, counts = scores_and_counts(context)
        ids = selection['ids']
        for i in ids:
            if (source_uid, i) in seen or i not in scores or scores[i] >= 4 or counts.get(i, 0) <= 0:
                raise ValueError('Duplicate or invalid approved class')
            seen.add((source_uid, i))
            if scores[i] < 3:
                strict_count += counts[i]
        job = experiment.project.find_job(selection['job'])
        if job.status != 'completed' or job.outputs['particles_selected'].num_items != sum(counts[i] for i in ids):
            raise ValueError('Approved native selection is incomplete or population differs')
        sources.append(job)
    if strict_count != gate['qualified_particles'] or strict_count < 2001:
        raise ValueError('Approved strict-good particle count differs from evidence')
    refine = experiment.project.find_job(outcome['best_refine'])
    if refine.status != 'completed' or not math.isfinite(experiment.resolution(refine)):
        raise ValueError('Approved preliminary refinement is unavailable')
    data = [job.load_output('templates_selected').filter_fields(lambda f: f == 'uid' or f.startswith('blob/'))
            for job in sources]
    bank = experiment.external(approval['tag']+'_manual_bank', 'template', 'templates',
        {'source_'+str(i): (job.uid, 'templates_selected') for i, job in enumerate(sources)},
        data[0].append(*data[1:]) if len(data) > 1 else data[0])
    accepted = dict(row, promotion_gate=gate, bank_job=bank.uid, best_refine=refine.uid,
                    selections=outcome['selections'], tag=approval['tag'])
    experiment.state.update(pilot_accepted=accepted, pilot_recipe=accepted, bank_job=bank.uid,
                            manual_diversity_approval=approval,
                            pre_approval_picker_parameters=experiment.state.get('picker_parameters'),
                            picker_parameters=copy.deepcopy(row['picker_parameters']))
    experiment.persist()
    return accepted
