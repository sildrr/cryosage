from types import SimpleNamespace

from dynamic_candidates import registry_parameter_template
from job_specs import get_parameter_template, annotate_extraction_parameters
from visual_decision_contract import STAGE_INSTRUCTIONS


def test_local_extraction_parameters_explain_absolute_output_size():
    for kind in ('extract_micrographs_multi', 'extract_micrographs_cpu_parallel'):
        field = get_parameter_template(kind)['bin_size_pix']
        assert field['type'] == 'integer'
        assert 'NOT a downsampling factor' in field['description']
        assert '2x2' in field['description']
        assert 'Nyquist' in field['description']
        assert 'default' not in field


def test_registry_annotation_keeps_native_constraints_and_defaults():
    original = {'bin_size_pix': {'type': 'integer', 'minimum': 2, 'multipleOf': 2,
                                'default': 128}}
    enriched = annotate_extraction_parameters('extract_micrographs_multi', original)
    assert enriched['bin_size_pix']['minimum'] == 2
    assert enriched['bin_size_pix']['multipleOf'] == 2
    assert enriched['bin_size_pix']['default'] == 128
    assert 'absolute OUTPUT EDGE LENGTH' in enriched['bin_size_pix']['description']


def test_dynamic_registry_exposes_semantics_and_other_jobs_are_unchanged():
    spec = SimpleNamespace(type='extract_micrographs_multi', params={
        'bin_size_pix': SimpleNamespace(type='integer', ge=2, default=None)})
    field = registry_parameter_template(spec)['bin_size_pix']
    assert field['minimum'] == 2
    assert 'NOT a downsampling factor' in field['description']
    assert annotate_extraction_parameters('other', {}) == {}


def test_visual_extraction_guidance_covers_optimization_and_unset():
    text = STAGE_INSTRUCTIONS['extract_box_size']
    assert 'OUTPUT EDGE LENGTH IN PIXELS' in text
    assert 'unset/blank' in text
    assert 'resolution optimization' in text
