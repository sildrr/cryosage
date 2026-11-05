import json
from types import SimpleNamespace

import pytest

from scripts import pilot_3d_diagnostic as diagnostic


def state():
    return dict(status='pilot_exhausted_no_validated_route',trials=[dict(after_job='J2286',
        objective_gate=dict(failed_conditions=['insufficient_supported_2d_shape_groups'],
                            promotion_class_ids=[3,4],qualified_particles=7429))])


def test_diagnostic_is_explicit_exception_not_gate_change(tmp_path,monkeypatch):
    import cryosparc_client
    monkeypatch.setattr(diagnostic,'PARENT',tmp_path)
    monkeypatch.setattr(diagnostic,'ROOT',tmp_path/'diagnostic')
    original = state()
    (tmp_path/'state.json').write_text(json.dumps(original))
    source = SimpleNamespace(uid='J2286',status='completed',type='class_2D_new')
    workspace = SimpleNamespace(find_jobs=lambda:[SimpleNamespace(uid='J2286')])
    project = SimpleNamespace(find_job=lambda uid:source,find_workspace=lambda uid:workspace)
    monkeypatch.setattr(cryosparc_client,'cryosparc_client',lambda:SimpleNamespace(find_project=lambda uid:project))
    diagnostic.prepare()
    manifest = json.loads((diagnostic.ROOT/'manifest.json').read_text())
    assert manifest['explicit_gate_exception']
    assert manifest['source_job']=='J2286' and manifest['expected_particles']==7429
    assert manifest['max_new_jobs']==4
    assert json.loads((tmp_path/'state.json').read_text())==original
    assert json.loads((diagnostic.ROOT/'benchmark_budget.json').read_text())['maximum']==5
    with pytest.raises(RuntimeError,match='duplicate'):
        diagnostic.prepare()


def test_diagnostic_does_not_silently_override_other_failed_gates(tmp_path,monkeypatch):
    monkeypatch.setattr(diagnostic,'PARENT',tmp_path)
    monkeypatch.setattr(diagnostic,'ROOT',tmp_path/'diagnostic')
    value=state()
    value['trials'][0]['objective_gate']['failed_conditions'].append('insufficient_cryosift_qualified_particles')
    (tmp_path/'state.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='diversity-only'):
        diagnostic.prepare()
