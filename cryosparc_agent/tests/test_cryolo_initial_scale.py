import copy
import json
from pathlib import Path

import pytest

from cryolo_initial_scale import CryoloInitialScaleMixin, snapshot_scale, validate_scale


def prior(dataset='10025', diameter=135.445, pixel=.6575, box=416):
    return dict(version='cryolo-initial-scale-v1', dataset=dataset, diameter_max_A=diameter,
                pixel_size_A=pixel, box_size_pix=box)


class Base:
    def model(self, slug, messages, model, validate):
        self.messages = messages
        return validate(copy.deepcopy(self.response))

    def stage(self, *args):
        return 'base stage'

    def review_parameters(self, *args):
        return 'base review'


class Controller(CryoloInitialScaleMixin, Base):
    def __init__(self, root, enabled=True):
        self.root = root
        self.manifest = dict(dataset='10025', acquisition_facts=dict(psize_A=.6575))
        if enabled:
            self.manifest['cryolo_initial_scale'] = prior()
        self.state = {}

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))


@pytest.mark.parametrize('dataset,diameter,pixel,box', [
    ('10025',135.445,.6575,416), ('10228',181.9759945869,1.058,352),
    ('10764',246.8199978471,.82,608), ('11280',168.6320012212,.856,400)])
def test_measured_snapshot_rounds_twice_diameter_up_without_masks(tmp_path, dataset, diameter, pixel, box):
    folder = tmp_path/dataset
    folder.mkdir()
    (folder/'summary.json').write_text(json.dumps(dict(dataset=dataset,
        statistics={'0.5': dict(count=100,median_A=diameter)},
        weights_sha256='a'*64, measurement='max estimated axis')))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(datasets={dataset:dict(
        images=[dict(pixel_size_A=pixel)])})))
    result = snapshot_scale(tmp_path,dataset,pixel)
    assert result['box_size_pix'] == box
    assert result['diameter_max_A'] == diameter
    assert 'mask' not in json.dumps(result) and 'reference_map' not in result
    assert len(result['source_summary_sha256']) == 64
    with pytest.raises(ValueError, match='pixel size'):
        snapshot_scale(tmp_path,dataset,pixel*2)


@pytest.mark.parametrize('value', [prior(box=440), prior(diameter=float('nan')), prior(diameter=1000,box=720)])
def test_no_silent_geometry_rewrite_or_clamp(value):
    with pytest.raises(ValueError):
        validate_scale(value,'10025',.6575)


def test_default_and_completed_ladder_passthrough(tmp_path):
    run = Controller(tmp_path,enabled=False)
    assert not run.initial_scale_locked()
    assert run.stage('s','extract_box_size',{}) == 'base stage'
    assert run.review_parameters('s','extract',{},dict(box_size_pix=440),[]) == 'base review'
    run.manifest['cryolo_initial_scale'] = prior()
    assert run.initial_scale_locked()
    run.state['initial_pilot_exhausted'] = True
    assert not run.initial_scale_locked()
    assert run.stage('s','extract_box_size',{}) == 'base stage'
    run.state = dict(pilot_accepted=dict(passed=True))
    assert not run.initial_scale_locked()


def test_initial_box_is_formula_record_not_fabricated_model_output(tmp_path):
    run = Controller(tmp_path)
    assert run.stage('pilot_r0_n50_box_parameters','extract_box_size',{'images': []}) == dict(box_size_pix=416)
    record = json.loads((tmp_path/'decisions/pilot_r0_n50_box_parameters/decision.json').read_text())
    assert record['decision_source'] == 'measured_formula_not_model_output'
    assert record['scale_estimate']['diameter_max_A'] == 135.445
    assert run.stage('pilot_r0_n50_box_parameters','extract_box_size',{'images': []}) == dict(box_size_pix=416)
    run.manifest['cryolo_initial_scale'] = prior(diameter=140,box=432)
    with pytest.raises(ValueError,match='Existing'):
        run.stage('pilot_r0_n50_box_parameters','extract_box_size',{})


def test_initial_blob_maximum_is_measured_but_minimum_is_selected_once(tmp_path):
    run = Controller(tmp_path)
    run.response = dict(parameters=dict(diameter=90,diameter_max=135.445))
    assert run.model('pilot_picker_parameters',[], 'model', lambda x:x)['parameters']['diameter'] == 90
    assert 'CONTROLLED INITIAL-SCALE' in run.messages[0]['content']
    run.response['parameters']['diameter_max'] = 200
    with pytest.raises(ValueError,match='maximum'):
        run.model('pilot_picker_parameters',[],'model',lambda x:x)


@pytest.mark.parametrize('target',['blob','extract'])
def test_scale_cannot_change_in_initial_diagnosis_but_unlocks_after_exhaustion(tmp_path,target):
    run = Controller(tmp_path)
    run.response = dict(target=target,parameters={})
    with pytest.raises(ValueError,match='only target=inspect'):
        run.model('pilot_r1_n50_diagnosis',[],'model',lambda x:x)
    run.state['initial_pilot_exhausted'] = True
    assert run.model('diameter_recovery_diagnosis',[],'model',lambda x:x)['target'] == target


def test_inspect_remains_model_controlled_and_extract_does_not_change(tmp_path):
    run = Controller(tmp_path)
    run.response = dict(target='inspect',parameters=dict(lpower_thresh_max=4))
    assert run.model('diagnosis',[],'model',lambda x:x)['target'] == 'inspect'
    assert run.review_parameters('e','inspect',{}, {},[]) == 'base review'
    assert run.review_parameters('e','extract',{},dict(box_size_pix=416),[])['action'] == 'keep'
    with pytest.raises(ValueError,match='geometry'):
        run.review_parameters('e','extract',{},dict(box_size_pix=440),[])
