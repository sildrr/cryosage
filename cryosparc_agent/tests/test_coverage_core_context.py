from types import SimpleNamespace

from coverage_rollout import CoverageRollout


def test_recovery_review_receives_real_core_source_and_does_not_require_new_core(tmp_path):
    import json
    instance=CoverageRollout.__new__(CoverageRollout)
    instance.root=tmp_path
    core=tmp_path/'core.json'
    core.write_text(json.dumps(dict(contact_sheet=dict(data_url='actual-core-image'))))
    instance.state=dict(current_core=dict(source=dict(project_uid='P6',job_uid='CORE'),
        class_ids=[7],context_file=str(core)))
    context=dict(source=dict(project_uid='P6',job_uid='CANDIDATE'),
        class_statistics=[dict(class_id=5,particle_count=500)],
        contact_sheet=dict(data_url='candidate-image'),
        cryosift=dict(success=True,project_uid='P6',job_uid='CANDIDATE',class_scores=[dict(class_id=5,score=4.035)]))
    captured=[]
    def model(slug,messages,name,validate):
        captured.extend(messages)
        return validate(dict(selected_class_ids=[5],reason='complements actual core'))
    instance.model=model
    assert instance.review('test',context,'recovered')==[5]
    content=captured[1]['content']
    assert any(c.get('image_url',{}).get('url')=='actual-core-image' for c in content)
    assert any('CORE' in c.get('text','') and 'EXISTING CORE' in c.get('text','') for c in content)
    assert 'Do NOT invent a new provisional core' in content[0]['text']
