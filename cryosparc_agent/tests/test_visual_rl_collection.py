import asyncio
import json
from types import SimpleNamespace

import pytest
import visual_rl_collection as visual_rl_module

from visual_rl_collection import (
    CollectionStore,
    collect_images,
    compute_class2d_feedback,
    durable_messages,
    ensure_writable_data_root,
    _execution_job_to_recover,
    _class2d_decision,
    _model_decision,
    _model_worker_project_dir,
    parameter_space_for_dataset,
    require_expected_stage_source,
    run_batch,
    to_v2_execution_decision,
    validate_collection_config,
    validate_ctf_source,
    validate_micrograph_subset,
    validate_stage_decision,
)
from visual_decision_contract import build_stage_decision_messages


def config():
    return {
        "experiment_id": "pilot-1",
        "datasets": {"EMPIAR-10025": "train", "11280": "heldout"},
        "parameter_space": {
            "blob_picker_diameter": {"diameter": [80, 250], "diameter_max": [120, 350]},
            "inspect_picks_threshold": {
                "ncc_score_min": [0, 1], "local_power_min": [0, 20], "local_power_max": [0, 30],
            },
            "extract_box_size": {"box_size_pix": [320, 384, 440, 504]},
        },
        "limits": {"max_trials_per_dataset": 2, "max_trials_total": 4,
                   "max_jobs_per_trial": 2, "max_decisions_per_stage": 1,
                   "max_input_particles": 10000},
    }


def test_trial_ledger_enforces_budgets_and_dataset_level_splits(tmp_path):
    store = CollectionStore(tmp_path / "collection.sqlite")
    store.initialize_experiment(config())
    train_trial = store.create_trial("pilot-1", "10025")
    heldout_trial = store.create_trial("pilot-1", "11280")
    store.record_event(train_trial, "blob_picker_diameter", "model_decision", {
        "decision": {"decision": "blob_picker_gpu", "parameters": {"diameter": 150}},
    })
    with pytest.raises(ValueError, match="decision budget exhausted"):
        store.record_event(train_trial, "blob_picker_diameter", "model_decision", {})
    store.record_event(train_trial, "blob_picker_diameter", "job_status", {"status": "completed"}, "J1")
    store.record_event(train_trial, "inspect_picks_threshold", "job_status", {"status": "completed"}, "J2")
    with pytest.raises(ValueError, match="job budget exhausted"):
        store.record_event(train_trial, "extract_box_size", "job_status", {"status": "queued"}, "J3")
    store.record_outcome(train_trial, {"resolution_reward": -3.2})
    store.record_outcome(heldout_trial, {"resolution_reward": -4.0})
    assert len(store.completed_feedback("pilot-1", split="train")) == 1
    heldout = store.completed_feedback("pilot-1", split="heldout")
    assert len(heldout) == 1 and heldout[0]["dataset_id"] == "11280"


def test_trial_creation_with_stable_id_is_idempotent(tmp_path):
    store = CollectionStore(tmp_path / "collection.sqlite")
    store.initialize_experiment(config())
    first = store.create_trial("pilot-1", "10025", trial_id="batch-trial-1")
    second = store.create_trial("pilot-1", "EMPIAR-10025", trial_id="batch-trial-1")
    assert first == second == "batch-trial-1"
    with pytest.raises(ValueError, match="another experiment/dataset"):
        store.create_trial("pilot-1", "11280", trial_id="batch-trial-1")


def test_trial_limit_normalizes_empiar_m_branch(tmp_path):
    cfg = config()
    cfg["datasets"] = {"10275": "train", "10275m": "train"}
    cfg["limits"]["max_trials_per_dataset"] = 1
    store = CollectionStore(tmp_path / "collection.sqlite")
    store.initialize_experiment(cfg)
    store.create_trial("pilot-1", "10275")
    with pytest.raises(ValueError, match="trial budget exhausted"):
        store.create_trial("pilot-1", "10275m")


def test_sampling_is_fixed_to_twenty_micrographs_and_legacy_particle_cap_is_removed():
    cfg = validate_collection_config(config())
    assert cfg["sampling"] == {
        "method": "cryosparc_exposure_sets_tool",
        "micrograph_count": 20,
        "randomize": True,
        "reuse_subset_across_trials": True,
    }
    assert "max_input_particles" not in cfg["limits"]
    cfg["sampling"]["micrograph_count"] = 19
    with pytest.raises(ValueError, match="fixed at 20"):
        validate_collection_config(cfg)


def test_dataset_specific_parameter_space_overrides_shared_defaults():
    cfg = config()
    cfg["datasets"] = {"10025": "train", "11022": "train"}
    cfg["dataset_parameter_space"] = {
        "EMPIAR-11022": {
            "extract_box_size": {"box_size_pix": [128, 160, 192, 256]},
        },
    }
    cfg = validate_collection_config(cfg)
    assert parameter_space_for_dataset(cfg, "11022")["extract_box_size"]["box_size_pix"] == [128, 160, 192, 256]
    assert parameter_space_for_dataset(cfg, "EMPIAR-10025")["extract_box_size"]["box_size_pix"] == [320, 384, 440, 504]


def test_exploration_feedback_keeps_same_dataset_and_excludes_heldout(tmp_path):
    store = CollectionStore(tmp_path / "collection.sqlite")
    cfg = config()
    store.initialize_experiment(cfg)
    baseline = store.create_trial("pilot-1", "10025")
    store.record_event(baseline, "blob_picker_diameter", "model_decision", {
        "decision": {"decision": "blob_picker_gpu", "parameters": {"diameter": 150}},
    })
    store.record_outcome(baseline, {"resolution_reward": -3.2})
    heldout = store.create_trial("pilot-1", "11280")
    store.record_outcome(heldout, {"resolution_reward": -2.0})
    cfg["experiment_id"] = "explore"
    cfg["exploration"] = {"feedback_experiment_ids": ["pilot-1"], "guidance": "Explore safely"}
    store.initialize_experiment(cfg)
    trial = store.trial(store.create_trial("explore", "10025"))
    context = visual_rl_module.exploration_context(store, cfg, trial, "blob_picker_diameter")
    assert [item["trial_id"] for item in context["same_dataset_prior_feedback"]] == [baseline]
    assert context["previous_stage_parameters"] == [{"diameter": 150}]
    trial = store.trial(store.create_trial("explore", "11280"))
    assert visual_rl_module.exploration_context(store, cfg, trial, "blob_picker_diameter") == {}


def test_exploration_plan_reuses_subsets_and_creates_three_rounds(tmp_path, monkeypatch):
    import prepare_visual_rl_exploration as planner

    monkeypatch.setattr(planner, "ROOT", tmp_path)
    cfg = config()
    cfg["datasets"] = {f"EMPIAR-{20000 + index}": "train" for index in range(100)}
    rows = [
        {"dataset_id": dataset, "project": "P19", "workspace": f"W{index}",
         "subset_job_uid": f"J{index}", "ctf_job_uid": f"J{index + 100}",
         "trial_id": f"baseline-{index}"}
        for index, dataset in enumerate(cfg["datasets"])
    ]
    source = tmp_path / "runs/preparation" / planner.BASELINE
    visual_rl_module.write_json(source / "collection_config.json", cfg)
    visual_rl_module.write_json(source / "batch_manifest.json", {
        "batch_id": planner.BASELINE, "trials": rows,
    })
    planner.prepare()
    planner.prepare()
    target = tmp_path / "runs/preparation" / planner.EXPERIMENT
    expanded = visual_rl_module.read_json(target / "collection_config.json")
    assert expanded["limits"]["max_trials_per_dataset"] == 3
    assert expanded["limits"]["max_trials_total"] == 300
    for round_index in range(1, 4):
        manifest = visual_rl_module.read_json(target / f"round_{round_index}_manifest.json")
        assert len(manifest["trials"]) == 100
        assert [item["subset_job_uid"] for item in manifest["trials"]] == [item["subset_job_uid"] for item in rows]
        assert all("trial_id" not in item for item in manifest["trials"])


def test_dataset_specific_parameter_space_rejects_unknown_datasets():
    cfg = config()
    cfg["dataset_parameter_space"] = {
        "11022": {"extract_box_size": {"box_size_pix": [128, 160]}},
    }
    with pytest.raises(ValueError, match="undeclared dataset"):
        validate_collection_config(cfg)


def test_micrograph_subset_requires_twenty_unique_uids_and_fingerprint():
    subset = {
        "source_job_uid": "J44",
        "output_name": "split_0",
        "micrograph_uids": list(range(20)),
    }
    validated = validate_micrograph_subset(subset)
    assert validated["micrograph_count"] == 20
    assert validated["randomized_once_and_reused"] is True
    with pytest.raises(ValueError, match="exactly 20"):
        validate_micrograph_subset({**subset, "micrograph_uids": list(range(19))})
    with pytest.raises(ValueError, match="unique"):
        validate_micrograph_subset({**subset, "micrograph_uids": [0] * 20})
    with pytest.raises(ValueError, match="fingerprint"):
        validate_micrograph_subset({**subset, "fingerprint": "wrong"})
    ctf_bound = validate_micrograph_subset({**subset, "ctf_job_uid": "J45"})
    assert ctf_bound["ctf_job_uid"] == "J45"
    assert ctf_bound["ctf_output_name"] == "exposures"


def test_live_stage_sources_must_follow_the_same_subset_trial_chain():
    subset = {
        "source_job_uid": "J_SUBSET",
        "output_name": "split_0",
        "micrograph_uids": [str(index) for index in range(20)],
    }
    trial = {
        "metadata": {"fixed_micrograph_subset": validate_micrograph_subset(subset)},
        "events": [
            {"stage": "blob_picker_diameter", "event_type": "job_status", "job_uid": "J_BLOB",
             "payload": {"status": "completed"}},
            {"stage": "inspect_picks_threshold", "event_type": "job_status", "job_uid": "J_INSPECT",
             "payload": {"status": "completed"}},
        ],
    }
    require_expected_stage_source(trial, "blob_picker_diameter", "J_SUBSET")
    require_expected_stage_source(trial, "inspect_picks_threshold", "J_BLOB")
    require_expected_stage_source(trial, "extract_box_size", "J_INSPECT")
    with pytest.raises(ValueError, match="bound CTF/exposure"):
        require_expected_stage_source(trial, "blob_picker_diameter", "J_OTHER")
    with pytest.raises(ValueError, match="must continue from"):
        require_expected_stage_source(trial, "inspect_picks_threshold", "J_OTHER")


def test_blob_picker_must_start_from_ctf_when_subset_is_ctf_bound():
    trial = {"metadata": {"fixed_micrograph_subset": {
        "source_job_uid": "J_SUBSET", "ctf_job_uid": "J_CTF",
    }}, "events": []}
    require_expected_stage_source(trial, "blob_picker_diameter", "J_CTF")
    with pytest.raises(ValueError, match="bound CTF/exposure"):
        require_expected_stage_source(trial, "blob_picker_diameter", "J_SUBSET")


def test_ctf_source_requires_completed_job_and_exact_subset_count():
    class Dataset:
        def rows(self):
            return [{"uid": f"u{i}"} for i in range(18)]

    class Job:
        status = "completed"
        type = "patch_ctf_estimation_multi"
        model = SimpleNamespace(workspace_uids=["W1"])

        def load_output(self, output_name):
            assert output_name == "exposures"
            return Dataset()

    project = SimpleNamespace(find_job=lambda uid: Job())
    assert validate_ctf_source(project, "J_CTF", "W1", [f"u{i}" for i in range(20)]).status == "completed"
    with pytest.raises(ValueError, match="workspace"):
        validate_ctf_source(project, "J_CTF", "W2", [f"u{i}" for i in range(20)])


def test_failed_enqueue_execution_is_not_recovered_as_a_queued_job():
    failed = {
        "stage": "blob_picker_diameter",
        "event_type": "execution",
        "payload": {"dry_run": False, "result": {
            "success": True,
            "execution_result": {"execution_results": [{"job_uid": "J85", "success": False}]},
        }},
    }
    successful = {
        "stage": "blob_picker_diameter",
        "event_type": "execution",
        "payload": {"dry_run": False, "result": {
            "success": True,
            "execution_result": {"execution_results": [{"job_uid": "J90", "success": True}]},
        }},
    }
    assert _execution_job_to_recover({"events": [failed]}, "blob_picker_diameter") is None
    assert _execution_job_to_recover({"events": [successful]}, "blob_picker_diameter")["job_uid"] == "J90"


def test_trial_binds_same_fixed_micrograph_subset(tmp_path):
    store = CollectionStore(tmp_path / "collection.sqlite")
    store.initialize_experiment(config())
    trial_id = store.create_trial("pilot-1", "10025")
    subset = {
        "source_job_uid": "J44",
        "output_name": "split_0",
        "micrograph_uids": list(range(20)),
    }
    bound = store.bind_micrograph_subset(trial_id, subset)
    assert store.trial(trial_id)["metadata"]["fixed_micrograph_subset"] == bound
    with pytest.raises(ValueError, match="different fixed micrograph subset"):
        store.bind_micrograph_subset(trial_id, {
            **subset, "micrograph_uids": list(range(1, 21)),
        })


def test_local_stage_decision_reuses_injected_persistent_worker():
    class FakeWorker:
        def __init__(self):
            self.requests = []

        def generate(self, request_id, messages, max_new_tokens, temperature):
            self.requests.append((request_id, messages, max_new_tokens, temperature))
            return {"raw_text": '{"decision":"blob_picker_gpu"}', "event": "generation"}

    worker = FakeWorker()
    args = SimpleNamespace(
        backend="local",
        model_worker=worker,
        max_new_tokens=64,
        temperature=0.0,
    )
    messages = [{"role": "user", "content": "choose"}]

    raw_text, generation = _model_decision(messages, args)

    assert raw_text == '{"decision":"blob_picker_gpu"}'
    assert generation["event"] == "generation"
    assert len(worker.requests) == 1
    assert worker.requests[0][1:] == (messages, 64, 0.0)


def test_explicit_retry_renews_decision_budget_without_erasing_history(tmp_path):
    store = CollectionStore(tmp_path / "collection.sqlite")
    store.initialize_experiment(config())
    trial_id = store.create_trial("pilot-1", "10025")
    store.record_event(trial_id, "blob_picker_diameter", "model_decision_invalid", {"error": "bad JSON"})
    with pytest.raises(ValueError, match="budget exhausted"):
        store.record_event(trial_id, "blob_picker_diameter", "model_decision", {})
    store.record_event(trial_id, "execution", "retry_started", {"decision_budget_renewed": True})
    store.record_event(trial_id, "blob_picker_diameter", "model_decision", {"decision": {}})
    trial = store.trial(trial_id)
    assert len(trial["events"]) == 3
    assert len(visual_rl_module.current_attempt_events(trial)) == 1
    with pytest.raises(ValueError, match="budget exhausted"):
        store.record_event(trial_id, "blob_picker_diameter", "model_decision", {})


def test_dead_local_worker_is_restarted_once(tmp_path):
    class Worker:
        stderr_path = tmp_path / "model_worker.stderr.log"
        calls = 0
        process = SimpleNamespace(poll=lambda: 1)

        def generate(self, *args):
            self.calls += 1
            if self.calls == 1:
                raise BrokenPipeError("dead worker")
            return {"raw_text": "{}"}

        def close(self):
            pass

        def start(self):
            self.started = True

    worker = Worker()
    args = SimpleNamespace(backend="local", model_worker=worker, max_new_tokens=64, temperature=0.2)
    text, result = _model_decision([], args)
    assert text == "{}" and result["worker_restarted"]
    assert worker.calls == 2 and worker.started
    assert worker.stderr_path.name == "model_worker.restart-1.stderr.log"


def test_collection_passes_scoped_submission_authorization(tmp_path, monkeypatch):
    import autonomous_mcp_closed_loop

    captured = []

    async def fake_call(session, name, args):
        captured.append(args)
        return {"success": True, "execution_result": {"execution_results": [
            {"success": True, "job_uid": "J1", "status": "queued"},
        ]}}

    monkeypatch.setattr(autonomous_mcp_closed_loop, "call_tool_json", fake_call)
    store = CollectionStore(tmp_path / "collection.sqlite")
    store.initialize_experiment(config())
    trial = store.create_trial("pilot-1", "10025")
    result = asyncio.run(visual_rl_module._submit_decision(
        None, store, trial, "blob_picker_diameter", {"decision_type": "forward"},
        project="P19", workspace="W1", current_node="J0", dataset_info={}, submit=True,
    ))
    assert result["job_uid"] == "J1"
    assert captured[-1]["allow_approval_required_create"] is True
    asyncio.run(visual_rl_module._submit_decision(
        None, store, trial, "blob_picker_diameter", {}, project="P19", workspace="W1",
        current_node="J0", dataset_info={}, submit=False,
    ))
    assert captured[-1]["allow_approval_required_create"] is False
    with pytest.raises(ValueError, match="Unapproved"):
        asyncio.run(visual_rl_module._submit_decision(
            None, store, trial, "delete_jobs", {}, project="P19", workspace="W1",
            current_node="J0", dataset_info={}, submit=True,
        ))


def test_exception_group_preserves_leaf_failure_reason():
    error = ExceptionGroup("outer", [RuntimeError("worker exited"), ExceptionGroup("inner", [ValueError("bad box")])])
    assert visual_rl_module.exception_leaves(error) == [
        {"type": "RuntimeError", "message": "worker exited"},
        {"type": "ValueError", "message": "bad box"},
    ]


@pytest.mark.parametrize("error,expected", [
    (RuntimeError("CryoSPARC connection failed: code 500"), True),
    (ExceptionGroup("wrapped", [RuntimeError("API HTTP/1.1 503")]), True),
    (RuntimeError("code 429 received error response"), True),
    (RuntimeError("code 401 received error response"), False),
    (RuntimeError("Job J514 has no particles/micrographs"), False),
    (ValueError("box_size_pix must be one of [320, 384]"), False),
    (RuntimeError("CUDA out of memory"), False),
])
def test_service_classifier_distinguishes_policy_and_auth_failures(error, expected):
    assert visual_rl_module.is_transient_service_error(error) is expected


def test_service_retry_keeps_current_trial_and_recovers(tmp_path, monkeypatch):
    calls = []
    waits = []
    item = {"trial_id": "t1", "dataset_id": "10025", "status": "running"}
    state = {"status": "running", "trials": [item]}
    path = tmp_path / "state.json"

    async def fake_trial(args):
        calls.append(args.trial_id)
        if len(calls) <= 2:
            raise ExceptionGroup("MCP wrapper", [RuntimeError("CryoSPARC code 500")])
        return {"status": "completed"}

    async def fake_sleep(delay):
        saved = json.loads(path.read_text())
        assert saved["status"] == "waiting_service"
        assert saved["trials"][0]["status"] == "waiting_service"
        waits.append(delay)

    monkeypatch.setattr(visual_rl_module, "run_trial", fake_trial)
    monkeypatch.setattr(visual_rl_module.asyncio, "sleep", fake_sleep)
    args = SimpleNamespace(trial_id="t1", service_retry_delay_seconds=1,
                           service_retry_max_delay_seconds=3, service_retry_max_wait_seconds=100)
    result = asyncio.run(visual_rl_module.run_trial_with_service_retries(args, state, item, path))
    assert result["status"] == "completed"
    assert calls == ["t1", "t1", "t1"] and waits == [1, 2]
    assert state["status"] == "running" and "service_wait" not in state


def test_service_deadline_preserves_pending_checkpoint(tmp_path, monkeypatch):
    async def fake_trial(args):
        raise RuntimeError("CryoSPARC code 500")

    monkeypatch.setattr(visual_rl_module, "run_trial", fake_trial)
    item = {"trial_id": "t1", "dataset_id": "10025", "status": "running"}
    state = {"status": "running", "trials": [item]}
    path = tmp_path / "state.json"
    args = SimpleNamespace(service_retry_max_wait_seconds=0)
    with pytest.raises(visual_rl_module.CollectionServiceUnavailable):
        asyncio.run(visual_rl_module.run_trial_with_service_retries(args, state, item, path))
    assert json.loads(path.read_text())["status"] == "service_unavailable"
    assert item["status"] == "waiting_service"


def test_model_worker_script_directory_is_the_agent_package_directory():
    script_dir = _model_worker_project_dir()
    assert script_dir.name == "cryosparc_agent"
    assert (script_dir / "scripts" / "model_generate_vision_once.py").is_file()


def test_batch_plan_is_stable_and_requires_same_subset_per_dataset(tmp_path):
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "batch.json"
    config_path.write_text(json.dumps(config()), encoding="utf-8")
    manifest = {
        "batch_id": "pilot-batch",
        "trials": [
            {"dataset_id": "10025", "project": "P19", "workspace": "W5", "subset_job_uid": "J1", "ctf_job_uid": "JCTF1"},
            {"dataset_id": "10025", "project": "P19", "workspace": "W6", "subset_job_uid": "J1", "ctf_job_uid": "JCTF2"},
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(
        data_root=str(tmp_path / "data"),
        config=str(config_path),
        batch_manifest=str(manifest_path),
        submit=False,
    )

    first = asyncio.run(run_batch(args))
    second = asyncio.run(run_batch(args))

    assert first["status"] == "planned"
    assert [row["trial_id"] for row in first["trials"]] == [
        row["trial_id"] for row in second["trials"]
    ]
    manifest["trials"][1]["subset_job_uid"] = "J2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="must reuse one Exposure Sets Tool Job"):
        asyncio.run(run_batch(args))


def test_run_batch_sequentially_shares_worker_and_persists_completion(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "batch.json"
    config_path.write_text(json.dumps(config()), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "batch_id": "worker-reuse-batch",
        "trials": [
            {"dataset_id": "10025", "project": "P19", "workspace": "W5", "subset_job_uid": "J1", "ctf_job_uid": "JCTF1"},
            {"dataset_id": "10025", "project": "P19", "workspace": "W6", "subset_job_uid": "J1", "ctf_job_uid": "JCTF2"},
        ],
    }), encoding="utf-8")
    seen_workers = []

    class FakeWorker:
        requests_served = 6

        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    async def fake_run_trial(args):
        worker = args.model_worker_holder.setdefault("worker", FakeWorker())
        seen_workers.append(worker)
        return {"trial_id": args.trial_id, "status": "completed"}

    monkeypatch.setattr(visual_rl_module, "run_trial", fake_run_trial)
    args = SimpleNamespace(
        data_root=str(tmp_path / "data"),
        config=str(config_path),
        batch_manifest=str(manifest_path),
        submit=True,
    )

    result = asyncio.run(run_batch(args))

    state = json.loads((tmp_path / "data/runs/batches/worker-reuse-batch.json").read_text())
    assert result["status"] == "completed"
    assert len(seen_workers) == 2 and seen_workers[0] is seen_workers[1]
    assert seen_workers[0].close_calls == 1
    assert [trial["status"] for trial in state["trials"]] == ["completed", "completed"]


def test_batch_exports_initialization_failures_and_continues_without_worker(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "batch.json"
    config_path.write_text(json.dumps(config()))
    manifest_path.write_text(json.dumps({
        "batch_id": "failure-export-batch",
        "trials": [
            {"dataset_id": "10025", "project": "P19", "workspace": "W5", "subset_job_uid": "J1", "ctf_job_uid": "J2"},
            {"dataset_id": "11280", "project": "P19", "workspace": "W6", "subset_job_uid": "J3", "ctf_job_uid": "J4"},
        ],
    }))

    async def fake_run_trial(args):
        if args.dataset_id == "10025":
            raise ValueError("invalid model parameters")
        CollectionStore(tmp_path / "data/collection.sqlite").record_outcome(
            args.trial_id, {"resolution_reward": -4.0},
        )
        return {"status": "completed"}

    monkeypatch.setattr(visual_rl_module, "run_trial", fake_run_trial)
    args = SimpleNamespace(
        data_root=str(tmp_path / "data"), config=str(config_path),
        batch_manifest=str(manifest_path), submit=True, continue_on_trial_error=True,
    )
    result = asyncio.run(run_batch(args))
    state = json.loads((tmp_path / "data/runs/batches/failure-export-batch.json").read_text())
    episodes = [json.loads(line) for line in (tmp_path / "data/trajectories/pilot-1_train.jsonl").read_text().splitlines()]
    assert result["status"] == state["status"] == "completed_with_errors"
    assert [trial["status"] for trial in state["trials"]] == ["failed", "completed"]
    assert episodes[0]["outcome"]["episode_status"] == "failed"
    assert episodes[0]["outcome"]["failure"]["message"] == "invalid model parameters"
    assert episodes[0]["outcome"]["resolution_reward"] is None


def test_feedback_reports_weighted_resolution_and_coverage_guard():
    feedback = compute_class2d_feedback(
        [
            {"particle_count": 3000, "resolution_A": 3.0},
            {"particle_count": 1000, "resolution_A": 5.0},
            {"particle_count": 500, "resolution_A": None},
        ],
        input_particle_count=8000,
        min_usable_particles=3000,
        min_particle_coverage=0.5,
    )
    assert feedback["particle_weighted_resolution_A"] == 3.5
    assert feedback["particle_coverage"] == 0.5
    assert feedback["resolution_reward"] == -3.5
    assert feedback["eligible_for_policy_update"] is True


def test_low_particle_coverage_is_not_rewarded():
    feedback = compute_class2d_feedback(
        [{"particle_count": 100, "resolution_A": 2.2}],
        input_particle_count=10000,
    )
    assert feedback["resolution_reward"] is None
    assert not feedback["eligible_for_policy_update"]


def test_feedback_has_no_particle_count_upper_cap():
    feedback = compute_class2d_feedback(
        [{"particle_count": 25000, "resolution_A": 3.4}],
        input_particle_count=30000,
        min_usable_particles=1000,
        min_particle_coverage=0.5,
    )
    assert feedback["input_particle_count"] == 30000
    assert feedback["eligible_for_policy_update"] is True


def test_feedback_includes_cryosift_particle_retention_and_reference_metrics():
    feedback = compute_class2d_feedback(
        [
            {"class_id": 0, "particle_count": 700, "resolution_A": 3.0},
            {"class_id": 1, "particle_count": 300, "resolution_A": 4.0},
        ],
        input_particle_count=1000,
        reference_projection_metrics={
            "status": "completed",
            "particle_weighted_projection_consistency_ncc": 0.82,
            "view_diversity": {"normalized_view_entropy": 0.75},
        },
        cryosift_result={
            "success": True,
            "threshold": 3.0,
            "selected_class_ids": [1],
            "class_scores": [{"class_id": 0, "score": 4.1}, {"class_id": 1, "score": 2.0}],
        },
    )
    assert feedback["reference_projection"]["particle_weighted_projection_consistency_ncc"] == 0.82
    assert feedback["view_diversity"]["normalized_view_entropy"] == 0.75
    assert feedback["cryosift_retention"]["selected_particle_count"] == 300
    assert feedback["cryosift_retention"]["selected_particle_fraction"] == 0.3
    assert feedback["cryosift_particle_count_guard"]["passed"] is False
    assert feedback["eligible_for_policy_update"] is False


def test_empty_successful_cryosift_result_has_zero_retention():
    feedback = compute_class2d_feedback(
        [{"class_id": 0, "particle_count": 100, "resolution_A": 3.0}],
        input_particle_count=100,
        cryosift_result={"success": True, "selected_class_ids": [], "threshold": 3.0},
    )
    assert feedback["cryosift_retention"]["status"] == "completed"
    assert feedback["cryosift_retention"]["selected_particle_count"] == 0
    assert feedback["cryosift_retention"]["selected_particle_fraction"] == 0
    assert feedback["cryosift_particle_count_guard"]["passed"] is False


def test_stage_decision_must_stay_inside_configured_search_space():
    space = config()["parameter_space"]
    decision = {"decision": "extract_micrographs_multi", "parameters": {"box_size_pix": 384}}
    assert validate_stage_decision("extract_box_size", decision, space) is decision
    decision["parameters"]["box_size_pix"] = 720
    with pytest.raises(ValueError, match="one of"):
        validate_stage_decision("extract_box_size", decision, space)


def test_compact_inspect_decision_maps_to_cryosparc_v2_parameter_names():
    result = to_v2_execution_decision("inspect_picks_threshold", {
        "decision": "inspect_picks_v2",
        "parameters": {"ncc_score_min": 0.2, "local_power_min": 5, "local_power_max": 18},
    })
    assert result["decision_type"] == "forward"
    assert result["parameters"] == {
        "ncc_score_thresh": 0.2,
        "lpower_thresh_min": 5,
        "lpower_thresh_max": 18,
    }


def test_images_are_sent_to_model_but_persisted_as_paths():
    visual = {
        "contact_sheet": {"data_url": "data:image/png;base64,A", "local_path": "/tmp/a.png"},
        "evidence_images": [{"data_url": "data:image/png;base64,B", "local_path": "/tmp/b.png"}],
    }
    assert collect_images(visual) == ["data:image/png;base64,A", "data:image/png;base64,B"]
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,A"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,B"}},
    ]}]
    stored = durable_messages(messages, visual)
    assert [part.get("path") for part in stored[0]["content"][1:]] == ["/tmp/a.png", "/tmp/b.png"]
    assert "data:image" not in json.dumps(stored)


def test_visual_evidence_records_full_field_version_and_attachment_hash(tmp_path):
    import hashlib
    from visual_rl_collection import visual_evidence_record
    image = tmp_path / "full.png"
    image.write_bytes(b"png")
    record = visual_evidence_record({
        "visual_evidence_version": "full_field_v4",
        "evidence_images": [{"role": "full_micrograph_with_scale", "local_path": str(image),
                             "data_url": "data:image/png;base64,cG5n"}],
    })
    assert record["visual_evidence_version"] == "full_field_v4"
    assert record["attachments"][0]["sha256"] == hashlib.sha256(b"png").hexdigest()
    assert record["attachments"][0]["local_file_matches_attachment"] is True
    assert "data:image" not in json.dumps(record)
    image.write_bytes(b"changed")
    mismatch = visual_evidence_record({"contact_sheet": {
        "local_path": str(image), "data_url": "data:image/png;base64,cG5n",
    }})
    assert mismatch["visual_evidence_version"] == "legacy_unversioned"
    assert mismatch["attachments"][0]["local_file_matches_attachment"] is False


def test_images_can_be_reloaded_from_local_cache_without_storing_base64(tmp_path):
    image = tmp_path / "evidence.png"
    image.write_bytes(b"png")
    assert collect_images({"contact_sheet": {"local_path": str(image)}}) == [
        "data:image/png;base64,cG5n",
    ]


def test_data_root_is_created_and_class2d_defaults_are_fixed(tmp_path):
    root = ensure_writable_data_root(tmp_path / "collection")
    assert root.is_dir()
    decision = _class2d_decision({})
    assert decision["job_type"] == "class_2D_new"
    assert decision["parameters"] == {
        "class2D_K": 50, "compute_num_gpus": 1, "random_seed": 42,
    }


def test_inspect_prompt_explains_raw_cryosparc_power_threshold_scale():
    messages = build_stage_decision_messages(
        "inspect_picks_threshold",
        {"score_statistics": {"power": {"p50": 770.0}}},
    )
    prompt = messages[1]["content"][0]["text"]
    assert "raw `pick_stats/power` values" in prompt
    assert "not normalized scores, percentiles" in prompt
    assert "score_statistics.power" in prompt


def test_visual_prompt_surfaces_prior_parameter_validation_error():
    messages = build_stage_decision_messages(
        "blob_picker_diameter",
        {"previous_invalid_decisions": [{"error": "diameter_max outside bounds"}]},
    )
    prompt = messages[1]["content"][0]["text"]
    assert "previous response was rejected" in prompt
    assert "inside the configured search_space" in prompt
