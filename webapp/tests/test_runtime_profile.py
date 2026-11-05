import json

import pytest

from webapp import manage


def test_runtime_profile_is_account_scoped(monkeypatch, tmp_path):
    monkeypatch.setattr(manage.Path, "home", lambda: tmp_path)
    profile = tmp_path / ".config/cryosage/web-runtime.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({"CRYOAGENT_GPU_LANE": "h20_slurm", "CRYOAGENT_AUTO_SELECT_LANE": "0"}))
    monkeypatch.setenv("CRYOAGENT_GPU_LANE", "old")
    monkeypatch.setenv("CRYOAGENT_AUTO_SELECT_LANE", "1")
    manage.apply_runtime_profile()
    assert manage.os.environ["CRYOAGENT_GPU_LANE"] == "h20_slurm"
    assert manage.os.environ["CRYOAGENT_AUTO_SELECT_LANE"] == "0"


def test_absent_profile_preserves_other_account(monkeypatch, tmp_path):
    monkeypatch.setattr(manage.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("CRYOAGENT_AUTO_SELECT_LANE", "1")
    manage.apply_runtime_profile()
    assert manage.os.environ["CRYOAGENT_AUTO_SELECT_LANE"] == "1"


def test_profile_rejects_unrelated_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(manage.Path, "home", lambda: tmp_path)
    profile = tmp_path / ".config/cryosage/web-runtime.json"
    profile.parent.mkdir(parents=True)
    profile.write_text('{"PATH":"untrusted"}')
    with pytest.raises(ValueError):
        manage.apply_runtime_profile()


def test_api_profile_selects_provider_models_and_credential_file(monkeypatch, tmp_path):
    monkeypatch.setattr(manage.Path, "home", lambda: tmp_path)
    profile = tmp_path / ".config/cryosage/web-runtime.json"
    profile.parent.mkdir(parents=True)
    values = {
        "OPENAI_BASE_URL": "https://www.packyapi.ai/v1", "OPENAI_MODEL": "gpt-6.1-sol",
        "CRYOAGENT_FIXED_API_MODEL": "gpt-6.1-sol", "CRYOAGENT_API_ENV_FILE": "/private/api.env",
    }
    profile.write_text(json.dumps(values))
    monkeypatch.setenv("OPENAI_API_KEY", "old-provider-key")
    for key in values:
        monkeypatch.setenv(key, "old")
    manage.apply_runtime_profile()
    assert {key: manage.os.environ[key] for key in values} == values
    assert "OPENAI_API_KEY" not in manage.os.environ


def test_gpu_only_profile_preserves_inherited_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(manage.Path, "home", lambda: tmp_path)
    profile = tmp_path / ".config/cryoagent/web-runtime.json"
    profile.parent.mkdir(parents=True)
    profile.write_text('{"CRYOAGENT_GPU_LANE":"h20_slurm"}')
    monkeypatch.setenv("OPENAI_API_KEY", "other-account-key")
    manage.apply_runtime_profile()
    assert manage.os.environ["OPENAI_API_KEY"] == "other-account-key"
