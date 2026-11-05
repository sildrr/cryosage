from pathlib import PurePosixPath
from types import SimpleNamespace

from scripts.inventory_comparison_maps import identify_dataset, map_record, scope


def test_shared_project_name_does_not_override_dataset():
    p = SimpleNamespace(uid='P6', title='EMPIAR-10025 Agent')
    assert identify_dataset(p, SimpleNamespace(title='W121 full-particle cleanup comparison'), [], None) == '11280'
    assert identify_dataset(p, SimpleNamespace(title='W125 low-scale p05 validation'), [], None) == '11280'
    assert identify_dataset(p, SimpleNamespace(title='CryoSage 121802 repeat'), [], None) == '12180'


def test_separate_particle_count_control_from_codex_repeats():
    p = SimpleNamespace(uid='P33')
    assert scope(p, SimpleNamespace(uid='W2', title='particle count control'))[0].startswith('06_')
    assert scope(p, SimpleNamespace(uid='W1', title='Codex repeat02'))[0].startswith('01_')


def test_remote_project_dir_becomes_local_path_before_existence_check(tmp_path):
    class Data:
        def fields(self):
            return ['map_sharp/path', 'map/path']

        def __getitem__(self, key):
            return ['sharp.mrc' if key.startswith('map_sharp/') else 'raw.mrc']

    (tmp_path/'sharp.mrc').touch()
    job = SimpleNamespace(uid='J1', type='homo_refine_new', load_output=lambda *a: Data(),
        model=SimpleNamespace(model_dump=lambda: dict(spec=dict(outputs=dict(volume=dict(
            latest_summary_stats=dict(fsc_info=dict(radwn_final_A=3.))))))))
    result = map_record(SimpleNamespace(dir=PurePosixPath(str(tmp_path))), job)
    assert result['sharpened_exists']
    assert not result['unsharpened_exists']
    assert result['resolution_A'] == 3.
