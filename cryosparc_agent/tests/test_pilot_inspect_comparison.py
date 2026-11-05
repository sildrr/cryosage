from copy import deepcopy
from types import SimpleNamespace

import pytest

from pilot_inspect_comparison import (InspectComparisonPilot, comparison_rank, validate_options)
from pilot_objective_gate import GatePolicy


def options():
    return dict(options=[dict(name=name, reason='matched evidence', parameters=dict(
        ncc_score_thresh=ncc, lpower_thresh_min=lo, lpower_thresh_max=hi))
        for name, ncc, lo, hi in [('permissive', .2, 500, 1600),
                                ('balanced', .3, 600, 1300), ('strict', .5, 700, 1000)]])


def test_threshold_alternatives_are_finite_distinct_and_nested():
    assert validate_options(options())
    for field, value in [('ncc_score_thresh', .1), ('lpower_thresh_min', 400),
                         ('lpower_thresh_max', 1800), ('lpower_thresh_max', float('nan'))]:
        bad = options()
        bad['options'][1]['parameters'][field] = value
        with pytest.raises(ValueError):
            validate_options(bad)
    bad = options()
    bad['options'][1]['parameters'] = deepcopy(bad['options'][0]['parameters'])
    with pytest.raises(ValueError, match='distinct'):
        validate_options(bad)


def test_relative_ranking_does_not_promote_or_reward_tiny_samples():
    row = dict(scoring_available=True, particles=3000, weighted_score=3.5,
               gate=dict(passed=False, qualified_particles=0, supported_group_count=0))
    tiny = dict(row, particles=100, weighted_score=2.0)
    assert comparison_rank(row, 2000) > comparison_rank(tiny, 2000)
    assert not row['gate']['passed']
    assert comparison_rank(dict(row, scoring_available=False), 2000) is None
    assert comparison_rank(dict(row, weighted_score=float('nan')), 2000) is None


def test_frozen_parameters_are_copied_without_calling_model():
    instance = InspectComparisonPilot.__new__(InspectComparisonPilot)
    frozen = dict(picker=dict(diameter=120, diameter_max=200),
                  inspect=dict(ncc_score_thresh=.3, lpower_thresh_min=600, lpower_thresh_max=1300),
                  extract=dict(box_size_pix=384))
    instance.state = dict(inspect_comparison=dict(frozen_parameters=deepcopy(frozen), classification=dict(class2D_K=5)))
    for scale in (50, 100, 200):
        assert instance.stage(f'n{scale}_inspect', 'inspect_picks_threshold', {}) == frozen['inspect']
        assert instance.stage(f'n{scale}_picker', 'blob_picker_diameter', {}) == frozen['picker']
        assert instance.stage(f'n{scale}_extract', 'extract_box_size', {}) == frozen['extract']
    p = instance.stage('n100_inspect', 'inspect_picks_threshold', {})
    p['ncc_score_thresh'] = 999
    assert instance.state['inspect_comparison']['frozen_parameters'] == frozen
    assert instance.classify_parameters('n50_class_parameters', 3000, 300) == dict(class2D_K=5)


def test_compare_submits_sibling_jobs_before_wait_and_reuses_winner(monkeypatch, tmp_path):
    import vision_inputs
    import pilot_inspect_comparison as module
    for name in ('build_micrograph_preview_visual_context', 'build_pick_inspection_visual_context',
                 'build_extract_box_visual_context'):
        monkeypatch.setattr(vision_inputs, name, lambda *a, **k: {})
    monkeypatch.setattr(module, 'load_class_images', lambda *a: {})
    monkeypatch.setattr(module, 'evaluate_gate', lambda *a: dict(
        available=True, passed=False, supported_group_count=0, qualified_particles=0))
    instance = InspectComparisonPilot.__new__(InspectComparisonPilot)
    instance.root = tmp_path
    instance.project = SimpleNamespace(uid='P6')
    instance.state = dict(steps={}, trials=[])
    instance.policy = GatePolicy()
    instance.service = lambda name, f: f()
    instance.persist = lambda: None
    instance.write = lambda *args: None
    instance.model = lambda *args: options()
    instance.stage = lambda slug, kind, visual: (dict(diameter=120, diameter_max=200)
        if kind == 'blob_picker_diameter' else dict(box_size_pix=384))
    instance.classify_parameters = lambda *args, **kwargs: dict(class2D_K=5, class2D_num_full_iter=3)
    calls = []
    jobs = {}
    def submit(slug, kind, sources, params, wait=True):
        calls.append((slug, wait, deepcopy(params)))
        jobs.setdefault(slug, SimpleNamespace(uid=slug, outputs=dict(particles=SimpleNamespace(num_items=3000))))
        instance.state['steps'][slug] = dict(job_uid=slug, status='completed')
        return jobs[slug]
    instance.submit = submit
    def classes(slug, job):
        score = 3.3 if 'balanced' in job.uid else 4.0
        return dict(score_summary=dict(available=True, particle_weighted_mean=score))
    instance.classes = classes
    instance.compare_once()
    c = instance.state['inspect_comparison']
    assert c['winner'] == 'balanced'
    assert c['frozen_parameters']['inspect'] == options()['options'][1]['parameters']
    for suffix in ('inspect', 'extract', 'class'):
        subset = [(slug, wait) for slug, wait, p in calls if slug.endswith('_'+suffix)]
        assert [wait for slug, wait in subset] == [False]*3 + [True]*3
    # Ten native Jobs for comparison, plus two selection/reclassification Jobs
    # at 50 and six each at 100/200: the original 24-new-Job budget still fits.
    assert len(jobs) == 10
    assert instance.state['steps']['n50_class']['job_uid'] == 'compare_balanced_class'
    before = len(calls)
    instance.compare_once()
    assert len(calls) == before


def test_no_valid_scores_cannot_choose_winner(monkeypatch):
    row = dict(scoring_available=False, particles=3000, weighted_score=2.0, gate={})
    assert comparison_rank(row, 2000) is None
