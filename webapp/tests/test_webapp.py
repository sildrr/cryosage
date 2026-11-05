import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from webapp import core, server, worker


@pytest.fixture
def configured(tmp_path, monkeypatch):
    inputs = tmp_path / "images"
    inputs.mkdir()
    (inputs / "one.mrc").write_bytes(b"sample")
    monkeypatch.setenv("CRYOAGENT_WEB_INPUT_ROOTS", str(inputs))
    monkeypatch.setattr(core, "DATA", tmp_path / "data")
    monkeypatch.setattr(core, "LOGS", tmp_path / "logs")
    return {"input_path": str(inputs / "*.mrc"), "name": "test", "pixel_size_A": .6575,
            "voltage_kV": 300, "cs_mm": 2.7, "dose": 53, "target": 2.8,
            "symmetry": "D7", "external_consent": True}


def test_validated_inputs_and_defaults(configured):
    result = core.validate_request(configured)
    assert result["input_file_count"] == 1 and result["max_jobs"] == 40
    assert result["input_type"] == "micrographs" and result["symmetry"] == "D7"


def test_api_model_defaults_and_explicit_overrides(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("CRYOAGENT_FIXED_API_MODEL", raising=False)
    assert core.api_models() == {"planner": "openai/gpt-6.1-sol", "execution": "openai/gpt-6-luna"}
    monkeypatch.setenv("OPENAI_MODEL", "custom/planner")
    monkeypatch.setenv("CRYOAGENT_FIXED_API_MODEL", "custom/executor")
    assert core.api_models() == {"planner": "custom/planner", "execution": "custom/executor"}


@pytest.mark.parametrize("field,value", [("pixel_size_A", float('nan')), ("target", 0),
                                          ("symmetry", "D7; whoami"), ("max_jobs", 500),
                                          ("external_consent", False), ("input_type", "unknown")])
def test_invalid_request_rejected(configured, field, value):
    configured[field] = value
    with pytest.raises(ValueError):
        core.validate_request(configured)


def test_path_escape_and_symlink_rejected(configured, tmp_path):
    external = tmp_path / "outside.mrc"
    external.write_bytes(b"x")
    parent = core.Path(configured["input_path"]).parent
    (parent / "escape.mrc").symlink_to(external)
    with pytest.raises(ValueError):
        core.validate_request(configured)
    with pytest.raises(ValueError):
        core.job_path("../../etc/passwd")


def test_catalog_dataset_symlink_is_allowed_but_nested_escape_is_not(configured, tmp_path):
    catalog = core.Path(configured["input_path"]).parent
    mounted = tmp_path / "mounted/10025/data"
    mounted.mkdir(parents=True)
    (mounted / "one.mrc").write_bytes(b"sample")
    (catalog / "10025").symlink_to(mounted.parent, target_is_directory=True)
    configured["input_path"] = str(catalog / "10025/data/*.mrc")
    assert core.validate_request(configured)["input_file_count"] == 1
    outside = tmp_path / "unapproved.mrc"
    outside.write_bytes(b"other")
    (mounted / "escape.mrc").symlink_to(outside)
    with pytest.raises(ValueError, match="resolves outside"):
        core.validate_request(configured)


@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_symmetry_is_unknown_not_c1(configured, value):
    configured["symmetry"] = value
    result = core.validate_request(configured)
    assert result["symmetry"] is None
    assert worker.symmetry_dataset_fields(result) == {"symmetry_status": "unknown"}


def test_unknown_symmetry_enables_center_search():
    assert worker.symmetry_runner_flags({"symmetry_status": "unknown"}) == [
        "--proshade-symmetry", "--proshade-find-symmetry-center"]
    assert worker.symmetry_runner_flags({"symmetry": "D2"}) == []
    assert worker.symmetry_runner_flags({"symmetry": "C1"}) == []


def test_missing_symmetry_is_unknown_but_explicit_c1_is_known(configured):
    configured.pop("symmetry")
    assert core.validate_request(configured)["symmetry"] is None
    configured["symmetry"] = " c1 "
    result = core.validate_request(configured)
    assert worker.symmetry_dataset_fields(result) == {"symmetry": "C1"}
    configured["symmetry"] = " d2 "
    assert core.validate_request(configured)["symmetry"] == "D2"


def test_submit_calls_only_control_slurm_not_model(configured, monkeypatch):
    import model_direct_runner
    monkeypatch.setattr(model_direct_runner, "resolve_api_key", lambda: "not-a-real-key")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="12345\n")

    monkeypatch.setattr(core.subprocess, "run", fake_run)
    result = core.submit_task(configured)
    assert result["status"] == "queued" and result["slurm_job"] == "12345"
    assert calls[0][0] == "sbatch"
    assert "not-a-real-key" not in json.dumps(result)
    assert core.read_task(result["id"])["config"]["target"] == 2.8


def test_web_policy_disables_experimental_scale_pilot(monkeypatch):
    import workflow_policy
    monkeypatch.setenv('CRYOAGENT_UNATTENDED', '0')
    monkeypatch.setenv("CRYOAGENT_INITIAL_SCALE_CALIBRATION", "1")
    monkeypatch.setenv("CRYOAGENT_FIXED_INITIAL_FUNNEL", "0")
    worker.configure_execution_policy()
    assert workflow_policy.fixed_initial_funnel_enabled()
    assert not workflow_policy.initial_scale_calibration_enabled()
    assert os.environ['CRYOAGENT_UNATTENDED'] == '1'


def test_snapshot_only_ranks_completed_refinements(monkeypatch):
    import quality_policy
    import workflow_state

    nodes = [
        {"cryosparc_job_uid": uid, "job_type": kind, "status": status,
         "parent_job_uids": [], "test_resolution": resolution}
        for uid, kind, status, resolution in (
            ("J1", "class_2D_new", "completed", 1.0),
            ("J2", "homo_refine_new", "completed", 3.2),
            ("J3", "nonuniform_refine_new", "completed", 2.7),
            ("J4", "homo_refine_new", "failed", 1.5),
        )
    ]
    monkeypatch.setattr(workflow_state, "extract_workflow_state", lambda *a: {"nodes": nodes})
    monkeypatch.setattr(quality_policy, "assess_node", lambda n: {
        "resolution_evidence": {"values": {"fsc_resolution": n["test_resolution"]}}})
    result = worker.snapshot("P1", "W1", 2.8)
    assert result["best_refine"]["job_uid"] == "J3" and result["target_met"]


@pytest.mark.parametrize("evidence,expected", [
    ({"fsc_resolution": 2.7}, (2.7, "fsc_resolution")),
    ({"fsc_info.radwn_final_A": 3.9929733236985667, "fsc_info.radwn_nomask_A": 11.67},
     (3.9929733236985667, "fsc_info.radwn_final_A")),
    ({"fsc_info.radwn_final_A": 4, "fsc_info_best.radwn_final_A": 3.9},
     (4., "fsc_info.radwn_final_A")),
    ({"fsc_info.radwn_nomask_05_A": 5, "map/psize_A": .6575}, (None, None)),
    ({"fsc_resolution": float('nan'), "fsc_info.radwn_final_A": 4}, (4., "fsc_info.radwn_final_A")),
])
def test_refinement_resolution_uses_explicit_final_fsc(evidence, expected):
    assert worker.refinement_resolution(evidence) == expected


@pytest.fixture
def web_server(tmp_path, monkeypatch):
    # Local HTTP tests must not depend on the PC's external proxy tunnel.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    monkeypatch.setattr(urllib.request, "urlopen", opener.open)
    monkeypatch.setattr(server, "TOKEN", "test-web-token-which-is-not-real")
    monkeypatch.setattr(server, "DATA", tmp_path / "data")
    app = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=app.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{app.server_port}"
    app.shutdown()
    app.server_close()


def test_home_and_workflow_assets_load(web_server):
    for path in ("/", "/workflow", "/style.css", "/app.js", "/api/showcase"):
        with urllib.request.urlopen(web_server + path) as response:
            assert response.status == 200
            assert response.read()


def test_authentication_and_origin_rejected(web_server):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(web_server + "/api/tasks")
    assert error.value.code == 401
    request = urllib.request.Request(web_server + "/api/tasks", data=b"{}", headers={
        "Authorization": "Bearer test-web-token-which-is-not-real", "Origin": "https://untrusted.example"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == 403


def test_authenticated_stream_upload(web_server, tmp_path):
    request = urllib.request.Request(web_server + "/api/uploads?filename=one.mrc", data=b"microscopy-test",
                                     headers={"Authorization": "Bearer test-web-token-which-is-not-real"})
    with urllib.request.urlopen(request) as response:
        result = json.load(response)
    assert (tmp_path / "data/uploads" / result["group"] / "one.mrc").read_bytes() == b"microscopy-test"


def test_finished_controller_without_refine_is_not_success(configured):
    identifier = "a" * 32
    core.write(core.job_path(identifier) / "task.json", {
        "id": identifier, "status": "finished", "stop_reason": "model_call_error", "config": configured,
    })
    task = core.public_task(identifier)
    assert task["status"] == "needs_attention"
    assert "model_call_error" in task["error"]


def test_compute_proxy_translation_is_scoped(tmp_path, monkeypatch):
    from webapp import proxy
    secret = tmp_path / "proxy.token"
    secret.write_text("temporary-test-token")
    monkeypatch.setattr(proxy, "TOKEN_FILE", secret)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:18888")
    monkeypatch.setenv("HTTPS_PROXY", "http://existing.internal:8080")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,172.16.1.2")
    assert proxy.configure_compute_proxy()
    assert proxy.urlparse(proxy.os.environ["HTTP_PROXY"]).hostname == proxy.HOST
    assert proxy.os.environ["HTTPS_PROXY"] == "http://existing.internal:8080"
    assert proxy.os.environ["NO_PROXY"] == "127.0.0.1,172.16.1.2"


def test_proxy_requires_auth_and_rejects_lan_targets():
    import socket
    import base64
    from webapp.proxy import ProxyServer, Relay
    app = ProxyServer(("127.0.0.1", 0), Relay)
    app.secret = "test-relay-token"
    thread = threading.Thread(target=app.serve_forever, daemon=True)
    thread.start()
    try:
        for header, expected in ((b"", b"407"),
                                 (b"Proxy-Authorization: Basic " + base64.b64encode(b"cryosage:test-relay-token") + b"\r\n", b"403")):
            with socket.create_connection(app.server_address, timeout=5) as client:
                client.sendall(b"CONNECT 172.16.1.2:61000 HTTP/1.1\r\n" + header + b"\r\n")
                assert expected in client.recv(4096).split(b"\r\n")[0]
    finally:
        app.shutdown()
        app.server_close()
