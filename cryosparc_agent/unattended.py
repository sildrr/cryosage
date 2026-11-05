"""Explicit unattended authorization and bounded, workspace-scoped evidence recovery."""
import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

MAX_FAILED_RETRIES = 3
MAX_EVIDENCE_REQUESTS = 3
GUIDANCE = (
    'Unattended execution is explicitly authorized by the user. Failed-job recovery '
    'retries marked automatic_retry are already approved; never ask for human approval. '
    'The controller automatically retrieves current-workspace upstream visual evidence '
    'and failure diagnostics. A request_input is a read-only evidence request to the '
    'controller, not a request to pause for a person. Use the retrieved evidence to '
    'choose a legal exploratory/recovery action and explain uncertainty. Do not invent '
    'acquisition facts, images, particle boundaries, permission or successful results. '
    'If one image is missing, use other observed evidence or a legal small-subset trial. '
    'Preserve numeric/input validation, Job/compute budgets, existing branch lineage, '
    'pilot limits and explicit user stop/pause. If no safe legal progress is possible '
    'after automatic evidence retrieval, return stop with a concrete unrecoverable '
    'reason; never remain waiting for manual input. No destructive action, arbitrary '
    'shell command or cross-workspace reference is authorized by this policy.'
)


def enabled():
    return os.getenv('CRYOAGENT_UNATTENDED', '0') == '1'


def input_signature(node):
    return sorted((group, str(item.get('source_job_uid')), str(item.get('source_output')))
                  for group, items in (node.get('inputs') or {}).items()
                  if isinstance(items, list) for item in items if isinstance(item, dict))


def authorize_retry(candidate, node, state=None):
    if not enabled():
        return candidate
    failures = sum(n.get('status') in {'failed', 'killed'} and
                   n.get('job_type') == node.get('job_type') and
                   input_signature(n) == input_signature(node)
                   for n in (state or {}).get('nodes', [node]))
    candidate.update(approval_required=False, approval_reasons=[],
                     automatic_retry=dict(approved=True, maximum_retries=MAX_FAILED_RETRIES,
                                          observed_failures=failures, authorization='explicit_user_policy'))
    if failures > MAX_FAILED_RETRIES:
        candidate.update(available=False, blocked_by=['automatic_failed_retry_limit'])
    return candidate


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    tmp.replace(path)


def consume_request(output, project, workspace, current, decision):
    path = Path(output) / 'automatic_evidence_requests.json'
    state = json.loads(path.read_text()) if path.exists() else dict(project=project, workspace=workspace, requests={})
    if (state['project'], state['workspace']) != (project, workspace):
        raise ValueError('Automatic evidence ledger belongs to another workspace')
    requests = state['requests'].setdefault(str(current), [])
    requests.append(dict(requested_inputs=decision.get('requested_inputs'), reason=decision.get('reason')))
    save(path, state)
    return len(requests) <= MAX_EVIDENCE_REQUESTS


def workspace_facts(project, workspace, current, requested):
    """Native API read only; model-supplied Job IDs never authorize another workspace."""
    from cryosparc_client import cryosparc_client
    cs = cryosparc_client()
    project_obj = cs.find_project(project)
    jobs = {j.uid: j for j in project_obj.find_workspace(workspace).find_jobs()}
    if current not in jobs:
        raise ValueError('Current job is outside the assigned workspace')
    requested_ids = re.findall(r'\bJ\d+\b', json.dumps(requested))
    queue = [current, *(uid for uid in requested_ids if uid in jobs)]
    records, seen = [], set()
    while queue and len(records) < 6:
        uid = queue.pop(0)
        if uid in seen or uid not in jobs:
            continue
        seen.add(uid)
        data = jobs[uid].model.model_dump()
        if workspace not in data.get('workspace_uids', []):
            raise ValueError('Evidence job is outside the assigned workspace')
        records.append(dict(uid=uid, job_type=data['spec']['type'], status=data.get('status'),
                            parameters=data['spec'].get('params', {}), cluster_job_id=data.get('cluster_job_id'),
                            runtime_errors=[str(e) for e in data.get('errors_run') or []]))
        queue.extend(data.get('parents', []))
    diagnostics = []
    for record in records[:3]:
        if record['status'] not in {'failed', 'killed'}:
            continue
        events = cs.api.jobs.get_event_logs(project, record['uid'])
        text = [str(e.text)[:2000] for e in events if getattr(e, 'type', None) == 'text'][-12:]
        diagnostic = dict(job_uid=record['uid'], event_log=text)
        from workflow_state import extract_run_errors
        diagnostic['run_errors'] = extract_run_errors(jobs[record['uid']])
        cluster_id = str(record.get('cluster_job_id') or '')
        if re.fullmatch(r'\d+(?:_\d+)?', cluster_id):
            try:
                accounting = subprocess.run(['sacct', '-j', cluster_id, '-n', '-P',
                    '--format=JobID,State,ExitCode,Reason,Elapsed,ReqMem'],
                    capture_output=True, text=True, timeout=15)
                diagnostic['scheduler_accounting'] = accounting.stdout[-6000:]
            except (OSError, subprocess.TimeoutExpired) as exc:
                diagnostic['scheduler_accounting_unavailable'] = type(exc).__name__
        diagnostics.append(diagnostic)
    return dict(records=records, failure_diagnostics=diagnostics,
                rejected_job_ids=sorted(set(requested_ids) - set(jobs)))


def image_parts(result):
    attachments = []
    images = [result.get('contact_sheet') or {}, *(result.get('evidence_images') or [])]
    scatter = result.get('score_scatter') or {}
    images.append(scatter)
    seen = set()
    for item in images:
        url = item.get('data_url')
        if isinstance(url, str) and url.startswith('data:image/') and url not in seen:
            attachments.append(dict(data_url=url, role=item.get('role', 'requested_visual_evidence'),
                                    description=item.get('description')))
            seen.add(url)
    return attachments[:3]


async def gather(call, session, project, workspace, current, candidates, requested=None,
                 facts_loader=workspace_facts):
    from upstream_evidence import collect_current_pilot, metadata
    bundle = await collect_current_pilot(call, session, project, workspace, current, candidates)
    if requested is None:
        return bundle
    facts = await asyncio.to_thread(facts_loader, project, workspace, current, requested)
    result = dict(success=True, lineage=facts, attachments=[], tool_results=[])
    if bundle:
        result['current_pilot'] = metadata(bundle)
        result['attachments'].extend(bundle.get('attachments', []))
    for node in facts['records'][:4]:
        typ, uid = node['job_type'], node['uid']
        arguments = dict(project_uid=project, job_uid=uid)
        if typ in {'blob_picker_gpu', 'template_picker_gpu', 'inspect_picks_v2'}:
            tool = 'get_pick_inspection_visual_context'
            arguments['max_micrographs'] = 2
        elif typ == 'class_2D_new':
            tool = 'get_class_average_visual_context'
            arguments['max_classes'] = 0
        elif typ in {'homo_abinit', 'homo_refine_new', 'nonuniform_refine_new'}:
            tool = 'get_abinitio_class2d_consistency_context'
        elif typ in {'patch_ctf_estimation_multi', 'import_micrographs', 'patch_motion_correction_multi'}:
            tool = 'get_micrograph_preview_visual_context'
            arguments['max_micrographs'] = 2
        else:
            continue
        response = await call(session, tool, arguments)
        result['tool_results'].append(dict(tool=tool, job_uid=uid, result=metadata(response)))
        result['attachments'].extend(image_parts(response))
    unique = {}
    for item in result['attachments']:
        unique.setdefault(item['data_url'], item)
    result['attachments'] = list(unique.values())[:8]
    return result
