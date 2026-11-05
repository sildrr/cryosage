"""Read-only analysis of the October 9 fixed-subset collection ledger."""

import collections
import json
import math
import sqlite3
import statistics
from pathlib import Path

from visual_rl_collection import write_json


ROOT = Path('/hdd1/linweifan/cryosparc/cryoagent_visual_rl')
BASE = 'visual-rl-100-collection-20261009'
EXPLORE = 'visual-rl-100-exploration-20261009'


def issue_codes(value):
    codes = set()
    if isinstance(value, dict):
        if isinstance(value.get('code'), str):
            codes.add(value['code'])
        for key, item in value.items():
            if key not in {'messages', 'raw_text', 'model_messages'}:
                codes.update(issue_codes(item))
    elif isinstance(value, list):
        for item in value:
            codes.update(issue_codes(item))
    return codes


def summary(values):
    values = [float(x) for x in values if isinstance(x, (float, int)) and math.isfinite(x)]
    return {'n': len(values), 'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def main():
    db = sqlite3.connect(f'file:{ROOT / "collection.sqlite"}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    batches = [BASE] + [f'{EXPLORE}-round-{i}' for i in range(1, 4)]
    results = []
    failure_counts = collections.Counter()
    failed_stages = collections.Counter()
    invalid_errors = collections.Counter()
    successful_by_dataset = collections.Counter()
    eligible_by_dataset = collections.Counter()
    signatures = collections.defaultdict(set)
    completed_signatures = collections.defaultdict(set)
    all_decisions = []
    same_subsets = collections.defaultdict(set)
    all_metrics = []
    examples = []
    for batch_id in batches:
        state = json.loads((ROOT / 'runs/batches' / f'{batch_id}.json').read_text())
        metrics = []
        reasons = collections.Counter()
        stage_counts = collections.Counter()
        for trial in state['trials']:
            dataset = trial['dataset_id']
            same_subsets[dataset].add((trial['subset_job_uid'], trial['ctf_job_uid']))
            decisions = {}
            failure = set()
            last_stage = 'initialization'
            last_event = None
            last_payload = {}
            for event in db.execute('SELECT stage,event_type,job_uid,payload_json FROM events WHERE trial_id=? ORDER BY event_id', (trial['trial_id'],)):
                last_stage = event['stage']
                last_event = event['event_type']
                payload = json.loads(event['payload_json'])
                last_payload = payload
                if event['event_type'] == 'model_decision':
                    decisions[event['stage']] = payload['decision']
                    all_decisions.append((event['stage'], payload['decision']['parameters']))
                if event['event_type'] == 'model_decision_invalid':
                    invalid_errors[str(payload.get('error'))] += 1
                if event['event_type'] == 'submission_failed':
                    failure.update(issue_codes(payload))
                    if not failure:
                        failure.add('submission_failed_without_issue_code')
                if event['event_type'] == 'job_status' and payload.get('status') in {'completed', 'failed', 'aborted'}:
                    stage_counts[f'{event["stage"]}:{payload["status"]}'] += 1
            signature = json.dumps(decisions, sort_keys=True)
            if len(decisions) == 3:
                signatures[dataset].add(signature)
            if trial['status'] == 'completed':
                completed_signatures[dataset].add(signature)
                successful_by_dataset[dataset] += 1
                feedback = trial['result']['feedback']
                metrics.append(feedback)
                all_metrics.append(feedback)
                if feedback['eligible_for_policy_update']:
                    eligible_by_dataset[dataset] += 1
                examples.append({
                    'dataset_id': dataset, 'trial_id': trial['trial_id'], 'batch': batch_id,
                    'decisions': decisions,
                    'weighted_resolution_A': feedback['particle_weighted_resolution_A'],
                    'cryosift_fraction': feedback['cryosift_retention'].get('selected_particle_fraction'),
                    'eligible': feedback['eligible_for_policy_update'],
                })
            else:
                if not failure:
                    if last_event == 'model_decision_invalid':
                        failure.add('model_invalid_after_retries')
                    elif last_event == 'job_status' and last_payload.get('status') == 'failed':
                        failure.add('cryosparc_job_failed')
                    elif last_event is None:
                        failure.add('before_first_decision_worker_failure_suspected')
                    else:
                        failure.add('after_job_before_next_decision_unresolved')
                for code in failure:
                    reasons[code] += 1
                    failure_counts[code] += 1
                failed_stages[last_stage] += 1
        results.append({
            'batch': batch_id, 'counts': dict(collections.Counter(t['status'] for t in state['trials'])),
            'failures': dict(reasons), 'terminal_job_status_events': dict(stage_counts),
            'eligible': sum(m['eligible_for_policy_update'] for m in metrics),
            'projection_completed': sum(m['reference_projection']['status'] == 'completed' for m in metrics),
            'cryosift_completed': sum(m['cryosift_retention']['status'] == 'completed' for m in metrics),
            'cryosift_empty': sum(m['cryosift_retention'].get('selected_class_count') == 0 for m in metrics),
            'resolution_A': summary([m['particle_weighted_resolution_A'] for m in metrics]),
            'cryosift_fraction': summary([m['cryosift_retention'].get('selected_particle_fraction') for m in metrics]),
        })
    histogram = collections.Counter(successful_by_dataset.get(dataset, 0) for dataset in same_subsets)
    parameters = collections.defaultdict(collections.Counter)
    for stage, values in all_decisions:
        for key, value in values.items():
            parameters[f'{stage}.{key}'][str(value)] += 1
    report = {
        'batches': results, 'attempted_trials': 400, 'completed_trials': len(all_metrics),
        'failure_issue_counts': dict(failure_counts), 'failed_last_stage_counts': dict(failed_stages),
        'invalid_decision_errors': dict(invalid_errors),
        'datasets_by_completed_trial_count': dict(sorted(histogram.items())),
        'datasets_with_multiple_distinct_completed_strategies': sum(len(x) >= 2 for x in completed_signatures.values()),
        'datasets_with_multiple_eligible_trials': sum(x >= 2 for x in eligible_by_dataset.values()),
        'eligible_trials': sum(m['eligible_for_policy_update'] for m in all_metrics),
        'fixed_subset_consistent': all(len(x) == 1 for x in same_subsets.values()),
        'total_valid_model_decisions': len(all_decisions),
        'parameter_top_values': {key: count.most_common(6) for key, count in parameters.items()},
        'successful_examples': examples,
    }
    target = ROOT / 'runs/analysis/visual-rl-400-20261012'
    write_json(target / 'summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'successful_examples'}, indent=2))


if __name__ == '__main__':
    main()
