import pytest

from pilot_reclassification_experiment import class_count, selection, next_step, stats, native_thresholds


@pytest.mark.parametrize('particles,expected', [(100,5),(3491,6),(5594,11),(16953,33),(233194,50)])
def test_particle_count_controls_class_granularity(particles, expected):
    assert class_count(particles) == expected


def test_sparse_pilot_cannot_start_classification():
    with pytest.raises(ValueError):
        class_count(0)


def test_selection_is_source_scoped_and_can_explicitly_be_empty():
    assert selection(dict(quality='poor', selected_class_ids=[], reason='No credible subset'), {1,2})
    for response in [dict(quality='usable', selected_class_ids=[], reason='x'),
                     dict(quality='uncertain', selected_class_ids=[3], reason='x'),
                     dict(quality='uncertain', selected_class_ids=[True], reason='x'),
                     dict(quality='uncertain', selected_class_ids=[1,1], reason='x')]:
        with pytest.raises(ValueError):
            selection(response, {1,2})


def test_expansion_is_bounded_and_never_forces_full_data():
    assert next_step('poor', 50000, 20) == 'expand'
    assert next_step('usable', 500, 100) == 'expand'
    assert next_step('usable', 5000, 100) == 'pilot_usable_for_promotion'
    assert next_step('uncertain', 50000, 200) == 'pilot_exhausted_no_validated_route'


def test_cryosift_weighted_statistics_and_failure_are_distinct():
    s=stats(dict(success=True, threshold=3., class_scores=[dict(class_id=0, score=2.),dict(class_id=1,score=4.)]), {0:100,1:300})
    assert s['particle_weighted_mean'] == 3.5
    assert s['recommended_particle_fraction'] == .25
    assert stats(dict(success=False), {})['available'] is False


def test_current_visual_threshold_contract_maps_to_native_names():
    p=native_thresholds(dict(ncc_score_min=.2, local_power_min=700, local_power_max=1500))
    assert p['ncc_score_thresh']==.2 and p['lpower_thresh_max']==1500
    assert p['calibrate_ncc'] is False
    with pytest.raises(ValueError, match='Conflicting'):
        native_thresholds(dict(ncc_score_min=.2,ncc_score_thresh=.3,local_power_max=1500))


def test_stage_uses_internal_kb_name_not_mcp_wrapper_name(monkeypatch, tmp_path):
    from pilot_reclassification_experiment import PilotExperiment
    import kb_bridge
    calls=[]
    monkeypatch.setattr(kb_bridge, 'call_kb_tool', lambda name, args: calls.append(name) or {})
    instance=PilotExperiment.__new__(PilotExperiment)
    instance.root=tmp_path
    instance.service=lambda name, function: function()
    instance.write=lambda *args: None
    instance.model=lambda slug, messages, model, validate: validate(dict(parameters=dict(diameter=100,diameter_max=200)))
    assert instance.stage('blob','blob_picker_diameter',{})['diameter']==100
    assert calls == ['get_job_doc']


@pytest.mark.parametrize('kind,sources,params', [
    ('blob_picker_gpu', {'micrographs':('J1','exposures')}, dict(diameter=100.,diameter_max=200.,num_process=20)),
    ('inspect_picks_v2', {'micrographs':('J2','micrographs'),'particles':('J2','particles')},
        dict(ncc_score_thresh=.2,lpower_thresh_min=700.,lpower_thresh_max=1500.,calibrate_ncc=False,calibrate_pow=False,do_auto_cluster=False)),
    ('extract_micrographs_multi', {'micrographs':('J3','micrographs'),'particles':('J3','particles')},
        dict(box_size_pix=384,num_extract=20,compute_num_gpus=1)),
    ('class_2D_new', {'particles':('J4','particles')},dict(class2D_K=10,random_seed=26026,compute_num_gpus=1)),
    ('select_2D', {'particles':('J5','particles'),'templates':('J5','class_averages')},dict(selected_templates='1,3')),
])
def test_all_experimental_steps_use_the_existing_validated_executor(kind,sources,params):
    from pilot_reclassification_experiment import PilotExperiment
    from v2_decision_adapter import adapt_v2_decision_to_internal
    from action_registry import execute_model_decision_payload
    instance = PilotExperiment.__new__(PilotExperiment)
    c = instance.candidate('test',kind,sources,params)
    d = dict(schema_version='2.0',decision_type='forward',job_type=kind,action=kind,
             action_id=c['action_id'],parameters=params)
    adapted=adapt_v2_decision_to_internal(d,[c])
    result=execute_model_decision_payload(adapted['internal_decision'],[c],dry_run=True)
    assert result['success'], result.get('issues')
    assert result['execution_plan']['actions'][0]['queue']['will_queue']
