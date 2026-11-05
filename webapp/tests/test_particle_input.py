"""Particle backend tests: all CryoSPARC and runner operations are mocked."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from webapp import core, worker


def star_text(*references):
    return ("data_optics\nloop_\n_rlnOpticsGroup #1\n_rlnVoltage #2\n1 300\n"
            "data_particles\nloop_\n_rlnImageName #1\n_rlnOpticsGroup #2\n" +
            "".join(f"'{reference}' 1\n" for reference in references))


@pytest.fixture
def particles(tmp_path, monkeypatch):
    folder = tmp_path / "inputs"
    folder.mkdir()
    (folder / "one.mrcs").write_bytes(b"stack")
    (folder / "two.mrcs").write_bytes(b"stack")
    star = folder / "particles.star"
    star.write_text(star_text("1@old/particles/one.mrcs", "2@old/particles/two.mrcs"))
    monkeypatch.setenv("CRYOAGENT_WEB_INPUT_ROOTS", str(folder))
    monkeypatch.setattr(core, "DATA", tmp_path / "tasks")
    return {"input_type": "particles", "input_path": str(folder), "particle_star_path": str(star),
            "pixel_size_A": 1.2, "voltage_kV": 300, "cs_mm": 2.7, "target": 3,
            "external_consent": True, "name": "particles"}


@pytest.mark.parametrize("glob_input", [False, True])
def test_folder_and_glob_with_optional_dose(particles, glob_input):
    if glob_input:
        particles["input_path"] += "/*.mrcs"
    config = core.validate_request(particles)
    assert config["input_file_count"] == 2
    assert config["particle_star_path"] == particles["particle_star_path"]
    assert config["dose"] is None


@pytest.mark.parametrize("value", [None, "", "relative.star", "/tmp/*.star", "/tmp/file.txt"])
def test_invalid_star_path(particles, value):
    particles["particle_star_path"] = value
    with pytest.raises(ValueError):
        core.validate_request(particles)


@pytest.mark.parametrize("reference,match", [
    ("1@missing.mrcs", "not selected"), ("1@../one.mrcs", "traversal"),
    ("0@one.mrcs", "positive index"), ("1@$HOME/one.mrcs", "environment"),
])
def test_bad_image_references(particles, reference, match):
    Path(particles["particle_star_path"]).write_text(star_text(reference))
    with pytest.raises(ValueError, match=match):
        core.validate_request(particles)


def test_absolute_reference_must_match_selected_stack(particles, tmp_path):
    star = Path(particles["particle_star_path"])
    star.write_text(star_text(f"1@{star.parent / 'one.mrcs'}"))
    assert core.validate_request(particles)["input_file_count"] == 2
    outside = tmp_path / "one.mrcs"
    outside.write_bytes(b"outside")
    star.write_text(star_text(f"1@{outside}"))
    with pytest.raises(ValueError, match="Absolute STAR"):
        core.validate_request(particles)


@pytest.mark.parametrize("which", ["star", "stack", "folder"])
def test_particle_symlink_escape(particles, tmp_path, which):
    folder = Path(particles["input_path"])
    if which == "folder":
        outside = tmp_path / "outside"
        outside.mkdir()
        (folder / "escape").symlink_to(outside, target_is_directory=True)
        particles["input_path"] = str(folder / "escape")
    else:
        target = folder / ("particles.star" if which == "star" else "one.mrcs")
        outside = tmp_path / target.name
        outside.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        core.validate_request(particles)


def test_ambiguous_star_basename_rejected(particles):
    Path(particles["particle_star_path"]).write_text(star_text("1@a/one.mrcs", "2@b/one.mrcs"))
    with pytest.raises(ValueError, match="Ambiguous"):
        core.validate_request(particles)


def test_duplicate_selected_basename_rejected(particles):
    folder = Path(particles["input_path"])
    for name in ("a", "b"):
        (folder / name).mkdir()
        (folder / name / "one.mrcs").write_bytes(b"stack")
    particles["input_path"] += "/*/*.mrcs"
    with pytest.raises(ValueError, match="basenames must be unique"):
        core.validate_request(particles)


def test_import_parameters_and_staging(particles, tmp_path):
    config = core.validate_request(particles)
    kind, params = worker.import_spec(config, tmp_path)
    assert kind == "import_particles"
    assert params["particle_meta_path"] == config["particle_star_path"]
    assert params["ignore_blob"] is False and params["output_constant_ctf"] is False
    assert not {"blob_paths", "total_dose_e_per_A2", "skip_header_check"}.intersection(params)
    staged = Path(params["particle_blob_path"])
    assert staged.is_dir() and not core.glob.has_magic(str(staged))
    assert sorted(p.name for p in staged.iterdir()) == ["one.mrcs", "two.mrcs"]
    assert all(p.is_symlink() and core.allowed_path(p.resolve()) for p in staged.iterdir())
    assert worker.import_spec(config, tmp_path) == (kind, params)
    (staged / "extra.mrcs").symlink_to(Path(config["input_path"]) / "one.mrcs")
    with pytest.raises(ValueError, match="unsafe resume"):
        worker.import_spec(config, tmp_path)


def test_fresh_particle_experiment_can_ignore_previous_3d_poses(particles, tmp_path):
    config = core.validate_request(particles)
    config['ignore_particle_pose'] = True
    _, params = worker.import_spec(config, tmp_path)
    assert params['ignore_pose'] is True
    assert params['output_constant_ctf'] is False
    assert params['ignore_blob'] is False


def output(*slots, count=2):
    return SimpleNamespace(type="particle", num_items=count,
                           results=[SimpleNamespace(name=slot) for slot in slots])


def test_class2d_connections_and_resume(particles, monkeypatch):
    task = {"id": "a" * 32, "config": core.validate_request(particles)}
    imported = SimpleNamespace(uid="J1", refresh=Mock(), outputs={
        "imported_particles": output("blob", "ctf"),
        "particles_incomplete": output("blob", "ctf"),
    })
    workspace = SimpleNamespace(create_job=Mock(return_value=SimpleNamespace(uid="J2")), find_job=Mock())
    worker.particle_class2d(workspace, imported, task)
    assert workspace.create_job.call_args.args == ("class_2D_new",)
    assert workspace.create_job.call_args.kwargs["connections"] == {"particles": ("J1", "imported_particles")}
    worker.particle_class2d(workspace, imported, task)
    workspace.create_job.assert_called_once()
    workspace.find_job.assert_called_once_with("J2")


def test_missing_ctf_blocks_class2d(particles):
    imported = SimpleNamespace(refresh=Mock(), outputs={"particles": output("blob")})
    workspace = SimpleNamespace(create_job=Mock())
    with pytest.raises(ValueError, match="CTF/optics"):
        worker.particle_class2d(workspace, imported, {"config": particles})
    workspace.create_job.assert_not_called()


def test_mocked_worker_handoff(particles, monkeypatch):
    from webapp import proxy, usage
    identifier = "b" * 32
    root = core.job_path(identifier)
    config = core.validate_request(particles)
    core.write(root / "task.json", {"id": identifier, "config": config})
    imported = SimpleNamespace(uid="J1", status='completed', refresh=Mock(), outputs={"imported_particles": output("blob", "ctf")})
    classified = SimpleNamespace(uid="J2", status='completed')
    workspace = SimpleNamespace(uid="W1", create_job=Mock(side_effect=[imported, classified]))
    project = SimpleNamespace(uid="P1", create_workspace=Mock(return_value=workspace))
    client = Mock(return_value=SimpleNamespace(find_project=Mock(return_value=project)))
    queue = Mock(return_value="completed")
    monkeypatch.setitem(sys.modules, "cryosparc_client", SimpleNamespace(cryosparc_client=client))
    monkeypatch.setitem(sys.modules, "prepare_visual_rl_batch", SimpleNamespace(queue_and_wait=queue))
    monkeypatch.setattr(proxy, "configure_compute_proxy", lambda: None)
    monkeypatch.setattr(worker, "configure_execution_policy", lambda: None)
    monkeypatch.setattr(worker.threading, "Thread", Mock())
    monkeypatch.setattr(worker, "snapshot", lambda *a: {"best_refine": None})
    monkeypatch.setattr(worker, "export_result", lambda *a: None)
    baseline = Mock(wraps=usage.initialize_usage)
    monkeypatch.setattr(usage, "initialize_usage", baseline)
    monkeypatch.setenv("CRYOAGENT_USAGE_LEDGER", "previous")

    def launch(command, **kwargs):
        assert os.environ["CRYOAGENT_USAGE_LEDGER"] == str(root / "api_usage.jsonl")
        baseline.assert_called_once_with(root)
        assert command[command.index("--current-node") + 1] == "J2"
        core.write(root / 'agent/run_mock/summary.json',
                   {'success': False, 'stop_reason': 'max_rounds_reached'})
        return SimpleNamespace(wait=lambda: None, poll=lambda: 0, returncode=0)

    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    worker.run(identifier)
    assert [call.args[0] for call in workspace.create_job.call_args_list] == ["import_particles", "class_2D_new"]
    assert [call.args[0].uid for call in queue.call_args_list] == ["J1", "J2"]
    dataset = json.loads((root / "dataset.json").read_text())
    assert dataset["input_type"] == "particles" and dataset["total_exposure_dose_e_per_A2"] is None
    assert set(dataset["available_input_files"]) == {"particle_blob_path", "particle_meta_path"}

