import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from webapp import pilot_pipeline as pipeline
from webapp import supervisor, worker
from webapp.core import write


def task(kind='micrographs'):
    return dict(id='a'*32, project='P20', workspace='W8', import_job='J1',
        model_config=dict(planner='provider/strong', execution='provider/cheap'),
        config=dict(input_type=kind, empiar_id='', pixel_size_A=1.23, voltage_kV=200,
                    cs_mm=2.7, dose=40., target=3.5, symmetry=None, kb=False, max_jobs=40))


def test_new_task_is_pinned_old_and_benchmark_unchanged():
    t = task()
    assert not pipeline.enabled(t)
    pipeline.pin(t, {})
    assert pipeline.enabled(t)
    assert t['config']['max_created_jobs'] == 100
    assert t['config']['decision_round_limit'] == 300
    other = task()
    other['config']['benchmark_profile'] = 'codex_20261017'
    pipeline.pin(other, {})
    assert 'pipeline_version' not in other


def test_particles_bypass_picking_but_keep_actual_budget():
    t = task('particles')
    pipeline.pin(t, dict(max_created_jobs=80))
    assert not pipeline.enabled(t)
    assert t['config']['max_created_jobs'] == 80


@pytest.mark.parametrize('value', [True, 0, 9, 201, 'invalid'])
def test_invalid_budgets(value):
    with pytest.raises(ValueError):
        pipeline.pin(task(), dict(max_created_jobs=value))


def test_generic_manifest_without_known_dataset_or_reference(tmp_path):
    t = task()
    pipeline.pin(t, {})
    m = pipeline.make_manifest(t, tmp_path, SimpleNamespace(uid='J3'), 73)
    assert m['dataset'] == t['id']
    assert m['pilot_scales'] == [50, 73]
    assert m['full_micrographs'] == 73
    assert m['acquisition_facts']['psize_A'] == 1.23
    assert m['gate']['duplicate_ncc'] == .9
    assert m['models'] == t['model_config']
    assert not m['kb_enabled']
    assert not any(k in m for k in ('reference_map', 'kb_snapshot', 'ablation_arm'))
    assert pipeline.make_manifest(t, tmp_path, SimpleNamespace(uid='J3'), 8)['pilot_scales'] == [8]
    with pytest.raises(ValueError):
        pipeline.make_manifest(t, tmp_path, SimpleNamespace(uid='J3'), 0)


def test_preprocessing_is_resumed_without_creating_again(tmp_path):
    workspace = Mock()
    workspace.create_job.return_value = SimpleNamespace(uid='J2')
    expected = SimpleNamespace(uid='J2')
    workspace.find_job.return_value = expected
    for _ in range(2):
        assert pipeline.create_preprocessing(task(), tmp_path, workspace, 'ctf', 'patch_ctf_estimation_multi',
            dict(compute_num_gpus=1), dict(micrographs=('J1', 'imported_micrographs'))) == expected
    assert workspace.create_job.call_count == 1
    with pytest.raises(ValueError, match='contract changed'):
        pipeline.create_preprocessing(task(), tmp_path, workspace, 'ctf', 'other', {})


def test_uncertain_preprocessing_never_blindly_retried(tmp_path):
    workspace = Mock()
    workspace.create_job.side_effect = RuntimeError('connection lost after creation')
    for _ in range(2):
        with pytest.raises(RuntimeError):
            pipeline.create_preprocessing(task(), tmp_path, workspace, 'ctf', 'patch_ctf_estimation_multi', {})
    assert workspace.create_job.call_count == 1


@pytest.mark.parametrize('kind', ['movies', 'micrographs'])
def test_preprocessing_uses_native_ctf_exposures_input(tmp_path, monkeypatch, kind):
    import prepare_visual_rl_batch
    monkeypatch.setattr(prepare_visual_rl_batch, 'queue_and_wait', lambda *a: 'completed')
    calls = []
    def create(t, root, workspace, key, job_type, params, connections):
        calls.append((key, job_type, params, connections))
        return SimpleNamespace(uid='J2' if key == 'motion' else 'J3', status='completed')
    monkeypatch.setattr(pipeline, 'create_preprocessing', create)
    pipeline.preprocessing(task(kind), tmp_path, Mock(), SimpleNamespace(uid='J1'))
    expected = ('J2', 'micrographs') if kind == 'movies' else ('J1', 'imported_micrographs')
    assert calls[-1][3] == {'exposures': expected}
    assert len(calls) == (2 if kind == 'movies' else 1)


def test_supervisor_accepts_only_recorded_pipeline_jobs(tmp_path):
    t = task()
    pipeline.pin(t, {})
    write(tmp_path/'preprocessing.json', dict(ctf=dict(job_uid='J3')))
    write(tmp_path/'state.json', dict(steps=dict(picker=dict(job_uid='J4'))))
    jobs = [dict(uid=j, status='building') for j in ['J1', 'J3', 'J4']]
    assert supervisor.recovery_checkpoint(tmp_path, t, [], jobs) is None
    with pytest.raises(ValueError, match='Uncheckpointed'):
        supervisor.recovery_checkpoint(tmp_path, t, [], jobs+[dict(uid='J5', status='building')])
    write(tmp_path/'manifest.json', dict(task_id=t['id'], project='P20', workspace='WRONG'))
    with pytest.raises(ValueError, match='identity'):
        supervisor.recovery_checkpoint(tmp_path, t, [], jobs)


def test_worker_uses_new_round_guard_without_changing_old_tasks(tmp_path, monkeypatch):
    monkeypatch.delenv('CRYOAGENT_WEB_SUPERVISED', raising=False)
    t = task()
    assert worker.supervised_resume(t, tmp_path) == ([], 40)
    pipeline.pin(t, {})
    assert worker.supervised_resume(t, tmp_path) == ([], 300)


def test_supervisor_accepts_pilot_records_alongside_first_planner_checkpoint(tmp_path):
    t = task()
    pipeline.pin(t, {})
    write(tmp_path/'preprocessing.json', dict(ctf=dict(job_uid='J3')))
    write(tmp_path/'state.json', dict(steps=dict(picker=dict(job_uid='J4'))))
    path = tmp_path/'agent/run_01/checkpoint.json'
    write(path, dict(project_uid=t['project'], workspace_uid=t['workspace'], current_node='J5'))
    jobs = [dict(uid=j, status='completed') for j in ['J1', 'J3', 'J4', 'J5']]
    assert supervisor.recovery_checkpoint(tmp_path, t, [], jobs) == path


def test_pipeline_resumes_manifest_and_hands_off_same_workspace(tmp_path, monkeypatch):
    import numpy as np
    from webapp import cryolo_probe
    t = task()
    pipeline.pin(t, {})
    ctf = Mock(uid='J3')
    ctf.load_output.return_value = np.array([(1.23,)] * 7, dtype=[('micrograph_blob/psize_A', 'f8')])
    pre = Mock(return_value=ctf)
    measure = Mock(return_value=None)
    monkeypatch.setattr(pipeline, 'preprocessing', pre)
    monkeypatch.setattr(cryolo_probe, 'estimate', measure)
    experiment = SimpleNamespace(run=lambda: 'J9', state=dict(chosen_route='blob'))
    monkeypatch.setattr(pipeline, 'picker_class', lambda: lambda root: experiment)
    workspace = Mock(uid=t['workspace'])
    workspace.find_jobs.return_value = [SimpleNamespace(uid='J1'), SimpleNamespace(uid='J9')]
    for _ in range(2):
        assert pipeline.run(t, tmp_path, SimpleNamespace(uid=t['project']), workspace, Mock()) == 'J9'
    assert pre.call_count == measure.call_count == 1
    context = json.loads((tmp_path/'coverage_planner_context.json').read_text())
    assert context['workspace_uid'] == t['workspace']
    assert context['best_refine'] == 'J9'


def test_cryolo_unavailable_is_explicit_not_a_fake_measurement(tmp_path, monkeypatch):
    from webapp import cryolo_probe
    monkeypatch.setattr(cryolo_probe, 'assets', lambda: (tmp_path/'python', tmp_path/'source', tmp_path/'weights'))
    assert cryolo_probe.estimate(task(), tmp_path, Mock(), Mock()) is None
    result = json.loads((tmp_path/'scale_probe/result.json').read_text())
    assert result['status'] == 'unavailable'
    assert result['fallback'] == 'model_visual_scale'
    assert 'initial_scale' not in result


def test_production_picker_supports_unknown_datasets_and_keeps_full_core_scope(tmp_path, monkeypatch):
    cls = pipeline.picker_class()
    from pilot_reclassification_experiment import PilotExperiment
    from two_tier_picker import TwoTierPicker
    monkeypatch.setattr(PilotExperiment, '__init__', lambda self, root: None)
    instance = cls.__new__(cls)
    t = task()
    pipeline.pin(t, {})
    instance.manifest = pipeline.make_manifest(t, tmp_path, SimpleNamespace(uid='J3'), 200)
    instance.state = dict(steps={})
    instance.root, instance.path, instance.write = tmp_path, tmp_path/'state.json', write
    instance.__init__(tmp_path)
    assert instance.policy.duplicate_ncc == .9
    assert instance.native_pixel_size_A == 1.23
    monkeypatch.setattr(TwoTierPicker, 'review', lambda *a, **kw: [2, 7])
    context = dict(source=dict(project_uid='P20', job_uid='J9'))
    assert instance.review('full_core_review', context, 'core') == [2, 7]
    assert instance.state['current_core']['source'] == context['source']
    assert instance.state['current_core']['class_ids'] == [2, 7]
    assert (tmp_path/'pilot_progress.json').is_file()


def test_pilot_honors_exact_provider_model_and_kb_off(tmp_path, monkeypatch):
    from pilot_reclassification_experiment import PilotExperiment
    import model_direct_runner
    import kb_bridge
    seen = []
    monkeypatch.setattr(model_direct_runner, 'resolve_api_key', lambda: 'dummy-test-key')
    def call(messages, base, key, model, **kwargs):
        seen.append(model)
        return dict(raw_text='{"parameters":{"diameter":80,"diameter_max":140}}')
    monkeypatch.setattr(model_direct_runner, 'run_openai_compatible_model', call)
    monkeypatch.setattr(kb_bridge, 'call_kb_tool', lambda *a, **kw: pytest.fail('KB-off must not query KB'))
    instance = SimpleNamespace(root=tmp_path, write=write,
        manifest=dict(production_pipeline=pipeline.VERSION, models=dict(planner='provider/exact-id'), kb_enabled=False),
        api_base='https://example.invalid/v1',
        acquisition_facts=dict(psize_A=1.23, accel_kv=200, cs_mm=2.7, total_dose_e_per_A2=40),
        service=lambda name, f: f())
    instance.model = lambda *args: PilotExperiment.model(instance, *args)
    assert PilotExperiment.stage(instance, 'diameter', 'blob_picker_diameter', {})['diameter_max'] == 140
    assert seen == ['provider/exact-id']
