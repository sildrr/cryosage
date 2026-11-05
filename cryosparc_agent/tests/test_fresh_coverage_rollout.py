from types import SimpleNamespace

import pytest

from coverage_rollout import CoverageRollout
from fresh_coverage_rollout import FreshCoverageRollout, validate_boxes, diameter_contract
from pilot_reclassification_experiment import PilotExperiment


@pytest.mark.parametrize('boxes', [[320, 384], [440, 504]])
def test_box_options(boxes):
    assert validate_boxes(dict(box_sizes_pix=boxes, reason='calibrated'))


@pytest.mark.parametrize('boxes', [[384, 384], [384, 400], [640, 728], [320., 384], [440, 384]])
def test_invalid_boxes(boxes):
    with pytest.raises(ValueError):
        validate_boxes(dict(box_sizes_pix=boxes, reason='bad'))


def test_measured_recipe_replaces_parent_fixed_values_only_after_pilot(monkeypatch):
    instance = FreshCoverageRollout.__new__(FreshCoverageRollout)
    instance.state = dict(pilot_recipe=dict(box_size_pix=512, picker_parameters=dict(diameter=90, diameter_max=180)))
    calls = []
    monkeypatch.setattr(CoverageRollout, 'submit', lambda self, slug, kind, sources, params, wait:
                        calls.append((slug, params)))
    instance.submit('grid_strict_384_extract', 'extract_micrographs_multi', {}, dict(box_size_pix=384))
    instance.submit('full_extract', 'extract_micrographs_multi', {}, dict(box_size_pix=440))
    instance.submit('confirm_blob_picker', 'blob_picker_gpu', {}, dict(diameter=120, diameter_max=220, num_process=50))
    assert calls[0][1]['box_size_pix'] == 384
    assert calls[1][1]['box_size_pix'] == 512
    assert calls[2][1] == dict(diameter=90, diameter_max=180, num_process=50)


def test_finished_run_never_replays_seed():
    instance = FreshCoverageRollout.__new__(FreshCoverageRollout)
    instance.state = dict(rollout_complete=True, best_refine='Jnew')
    instance.bank = lambda: pytest.fail('seed replay')
    assert instance.run() == 'Jnew'


def test_fresh_visuals_deduplicate_images():
    instance = FreshCoverageRollout.__new__(FreshCoverageRollout)
    visual = dict(contact_sheet=dict(data_url='data:image/png;base64,AA'),
                  evidence_images=[dict(data_url='data:image/png;base64,AA')])
    content = instance.visual_content('instruction', visual)
    assert len([i for i in content if i['type'] == 'image_url']) == 1


def test_fresh_class_does_not_inherit_seeded_bank_or_template_calibration():
    assert FreshCoverageRollout.bank is not CoverageRollout.bank
    assert FreshCoverageRollout.run is not CoverageRollout.run
    assert FreshCoverageRollout.template_parameters is not CoverageRollout.template_parameters


def test_diameter_request_does_not_receive_trailing_class_selection_policy(monkeypatch):
    instance = FreshCoverageRollout.__new__(FreshCoverageRollout)
    messages = [dict(role='system', content='diameter contract'),
                dict(role='user', content='Return parameters.diameter and diameter_max')]
    captured = []
    monkeypatch.setattr(PilotExperiment, 'model', lambda self, slug, msg, model, validate:
                        captured.extend(msg) or dict(parameters=dict(diameter=100, diameter_max=200)))
    instance.model('other_stage', messages, 'model', lambda x: x)
    assert captured == messages
    assert captured[-1]['content'] == 'Return parameters.diameter and diameter_max'


@pytest.mark.parametrize('content', ['generic stage', [dict(type='text', text='generic stage')]])
def test_controlled_pilot_requires_explicit_maximum_without_mutating_source(content):
    messages = [dict(role='user', content=content)]
    adjusted = diameter_contract(messages)
    final = adjusted[-1]['content']
    text = final if isinstance(final, str) else final[-1]['text']
    assert 'parameters.diameter_max' in text
    assert 'REQUIRED positive finite' in text
    assert 'uncertain' in text
    assert messages[-1]['content'] == content


def test_api4_model_alias_is_scoped_to_explicit_endpoint(monkeypatch):
    instance = FreshCoverageRollout.__new__(FreshCoverageRollout)
    instance.api_base = 'http://15.204.107.114:3009/v1'
    seen = []
    monkeypatch.setattr(PilotExperiment, 'model', lambda self, slug, msg, model, validate: seen.append(model))
    instance.model('thresholds', [], 'openai/gpt-6.1-sol', lambda x: x)
    instance.api_base = 'https://api.ofox.ai/v1'
    instance.model('thresholds', [], 'openai/gpt-6.1-sol', lambda x: x)
    assert seen == ['gpt-6.1-sol', 'openai/gpt-6.1-sol']


def test_api4_override_keeps_default_environment_and_rejects_unknown_routes(monkeypatch):
    import os
    from pathlib import Path
    from scripts.launch_fresh_coverage_rollout import configure_api_override
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://api.ofox.ai/v1')
    monkeypatch.setenv('CRYOAGENT_API_ENV_FILE', 'original')
    monkeypatch.setenv('OPENAI_API_KEY', 'placeholder-not-a-secret')
    configure_api_override({})
    assert os.environ['CRYOAGENT_API_ENV_FILE'] == 'original'
    with pytest.raises(ValueError):
        configure_api_override(dict(api_override=dict(base_url='http://unapproved')))
    configure_api_override(dict(api_override=dict(base_url='http://15.204.107.114:3009/v1',
        credential_file=str(Path.home()/'.config/cryoagent/api4.env'))))
    assert 'OPENAI_API_KEY' not in os.environ
    assert os.environ['OPENAI_MODEL'] == 'gpt-6.1-sol'
    assert os.environ['CRYOAGENT_API_ENV_FILE'].endswith('/api4.env')
