import pytest

from webapp.core import controller_command


def test_controller_qos(monkeypatch):
    monkeypatch.setenv("CRYOAGENT_WEB_SLURM_QOS", "gpu6")
    command = controller_command("test-task")
    assert "--qos=gpu6" in command
    assert command[-1] == "test-task"


def test_controller_partition_override(monkeypatch):
    monkeypatch.setenv("CRYOAGENT_WEB_SLURM_PARTITION", "g8m768")
    assert "--partition=g8m768" in controller_command("test-task")


def test_invalid_controller_partition(monkeypatch):
    monkeypatch.setenv("CRYOAGENT_WEB_SLURM_PARTITION", "g8m768 --other")
    with pytest.raises(ValueError, match="partition"):
        controller_command("test-task")


def test_default_qos_unchanged(monkeypatch):
    monkeypatch.delenv("CRYOAGENT_WEB_SLURM_QOS", raising=False)
    assert not any(arg.startswith("--qos") for arg in controller_command("test-task"))


def test_custom_script(monkeypatch, tmp_path):
    script = tmp_path / "controller.sbatch"
    script.write_text("#!/bin/bash\n")
    monkeypatch.setenv("CRYOAGENT_WEB_SLURM_SCRIPT", str(script))
    assert controller_command("test-task")[-2] == str(script)


def test_invalid_qos(monkeypatch):
    monkeypatch.setenv("CRYOAGENT_WEB_SLURM_QOS", "gpu6 --other")
    with pytest.raises(ValueError, match="QoS"):
        controller_command("test-task")
