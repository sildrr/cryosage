"""Slurm-side adapter; the existing agent is the only decision/execution engine."""
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from webapp.core import DATA, PYTHON, REPO, api_models, job_path, now, particle_input_files, read_task, write

sys.path.insert(0, str(REPO / "cryosparc_agent"))


def configure_execution_policy():
    from webapp.gpu_routing import configure
    configure()
    # Isolate the validated full-data funnel from later experimental policies.
    os.environ.update({
        "CRYOAGENT_UNATTENDED": "1",
        "CRYOAGENT_FIXED_INITIAL_FUNNEL": "1",
        "CRYOAGENT_INITIAL_SCALE_CALIBRATION": "0",
        "CRYOAGENT_ENABLE_BOX_SIZE_SWEEP": "0",
        "CRYOAGENT_ENABLE_QUALITY_ROLLBACK": "1",
    })


def symmetry_dataset_fields(config):
    symmetry = config.get("symmetry")
    if symmetry:
        return {"symmetry": symmetry}
    return {"symmetry_status": "unknown"}


def symmetry_runner_flags(dataset):
    if dataset.get("symmetry_status") == "unknown":
        return ["--proshade-symmetry", "--proshade-find-symmetry-center"]
    return []


def supervised_resume(task, root):
    """Resume only an identity-checked checkpoint, preserving the round ceiling."""
    path = task.get('service_resume_checkpoint')
    rounds = task['config'].get('decision_round_limit', task['config']['max_jobs'])
    if os.getenv('CRYOAGENT_WEB_SUPERVISED') == '1':
        rounds -= sum(1 for _ in (root / 'agent').glob('run_*/round_*') if _.is_dir())
    if not path:
        return [], rounds
    checkpoint = Path(path).resolve()
    if not checkpoint.is_relative_to((root / 'agent').resolve()):
        raise ValueError('Service checkpoint must belong to this task')
    data = json.loads(checkpoint.read_text())
    if (data.get('project_uid'), data.get('workspace_uid')) != (task['project'], task['workspace']):
        raise ValueError('Service checkpoint identity mismatch')
    return ['--resume-checkpoint', str(checkpoint)], rounds


def refinement_resolution(evidence):
    # Prefer the final FSC metric associated with the exported final map. Never
    # take the minimum over masks/0.5-cutoff values or unrelated resolution keys.
    for key in (
        "fsc_resolution", "fsc_info.radwn_final_A",
        "fsc_info.radwn_autotight_noisesub_A", "fsc_info.radwn_autotight_A",
        "fsc_info_best.radwn_final_A", "fsc_info_best.radwn_autotight_noisesub_A",
    ):
        value = evidence.get(key)
        if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            return float(value), key
    return None, None


def snapshot(project, workspace, target):
    from workflow_state import extract_workflow_state
    from quality_policy import assess_node
    state = extract_workflow_state(project, workspace)
    nodes = []
    candidates = []
    for node in state["nodes"]:
        evidence = assess_node(node).get("resolution_evidence", {}).get("values", {})
        resolution, resolution_metric = refinement_resolution(evidence)
        if isinstance(resolution, (float, int)) and math.isfinite(resolution) and resolution > 0 and node["job_type"] in {"homo_refine_new", "nonuniform_refine_new", "homo_refine"} and node["status"] == "completed":
            candidates.append({"job_uid": node["cryosparc_job_uid"], "resolution_A": resolution,
                               "resolution_metric": resolution_metric})
        nodes.append({"id": node["cryosparc_job_uid"], "type": node["job_type"],
                      "status": node["status"], "parents": node["parent_job_uids"],
                      "resolution_A": resolution, "errors": node.get("run_errors"),
                      "resolution_metric": resolution_metric,
                      "timestamps": node.get("timestamps", {}),
                      "parameters": node.get("key_parameters", {}),
                      "lane": node.get("runtime", {}).get("lane")})
    best = min(candidates, key=lambda c: c["resolution_A"]) if candidates else None
    return {"updated_at": now(), "nodes": nodes, "best_refine": best,
            "target_met": bool(best and best["resolution_A"] <= target)}


def export_result(task, progress):
    from cryosparc_client import cryosparc_client
    root = job_path(task["id"])
    best = progress.get("best_refine")
    result = {"status": "no_refine_result", "target_met": False, "artifacts": []}
    if best:
        project = cryosparc_client().find_project(task["project"])
        job = project.find_job(best["job_uid"])
        volume = job.load_output("volume")
        fields = set(volume.fields())
        field = "map_sharp/path" if "map_sharp/path" in fields else "map/path"
        remote = str(volume[field][0])
        output = root / "density.mrc"
        project.download_file(remote, output)
        result.update(status="available", **best, map_kind="sharpened" if "sharp" in field else "unsharpened",
                      source_map=remote, target_met=progress["target_met"], artifacts=["density.mrc"])
    report = ["# CryoSage reconstruction report", "", f"Dataset: {task['config']['name']}",
              f"CryoSPARC: {task['project']}/{task['workspace']}",
              f"Target: {task['config']['target']} A", f"Target met: {result['target_met']}",
              f"Best refinement: {result.get('job_uid', 'none')}",
              f"FSC resolution: {result.get('resolution_A', 'unavailable')} A", "",
              "A completed controller does not guarantee target resolution or biological correctness.",
              "Sharpened maps are preferred for downstream model building.", "", "## Acquisition and policy", ""]
    report.extend(f"- {key}: {task['config'][key]}" for key in
                  ("input_type", "input_file_count", "pixel_size_A", "voltage_kV", "cs_mm", "dose", "symmetry", "kb", "max_jobs"))
    if task.get('pipeline_version'):
        report.extend([f"- pipeline_version: {task['pipeline_version']}",
                       f"- actual_job_budget: {task['config']['max_created_jobs']}",
                       f"- decision_round_safety_limit: {task['config']['decision_round_limit']}"])
    report.extend(["", "## Jobs", ""])
    report.extend(f"- {n['id']} | {n['type']} | {n['status']} | resolution={n.get('resolution_A')}" for n in progress.get("nodes", []))
    (root / "report.md").write_text("\n".join(report))
    result["artifacts"].append("report.md")
    write(root / "result.json", result)


def monitor(task, stop):
    root = job_path(task["id"])
    while not stop.is_set():
        try:
            progress = snapshot(task["project"], task["workspace"], task["config"]["target"])
            write(root / "progress.json", progress)
            (root / "monitor_error.json").unlink(missing_ok=True)
        except Exception:
            write(root / "monitor_error.json", {"at": now(), "message": "CryoSPARC progress query failed; retrying"})
        stop.wait(30)


def import_spec(config, root):
    params = {"psize_A": config["pixel_size_A"], "accel_kv": config["voltage_kV"],
              "cs_mm": config["cs_mm"]}
    if config["input_type"] == "particles":
        files = particle_input_files(config["input_path"], config["particle_star_path"])
        # CryoSPARC searches suffixes of STAR paths within this directory. A flat
        # directory of unique, validated names supports relocated relative paths.
        stacks = root / "particle_stacks"
        stacks.mkdir(exist_ok=True)
        expected = {Path(f).name: Path(f).resolve() for f in files}
        for entry in stacks.iterdir():
            if entry.name not in expected or not entry.is_symlink() or entry.resolve() != expected[entry.name]:
                raise ValueError("Staged particle stacks changed; refusing unsafe resume")
        for name, target in expected.items():
            link = stacks / name
            if not link.is_symlink():
                link.symlink_to(target)
        params.update(particle_meta_path=config["particle_star_path"], particle_blob_path=str(stacks),
                      ignore_blob=False, output_constant_ctf=False)
        if config.get("ignore_particle_pose") is True:
            params["ignore_pose"] = True
        return "import_particles", params
    params.update(blob_paths=config["input_path"], total_dose_e_per_A2=config["dose"], skip_header_check=False)
    if config['input_type'] == 'movies' and config.get('gainref_path'):
        params['gainref_path'] = config['gainref_path']
    return "import_movies" if config["input_type"] == "movies" else "import_micrographs", params


def particle_class2d(workspace, imported, task):
    if task.get("particle_class2d_job"):
        return workspace.find_job(task["particle_class2d_job"])
    imported.refresh()
    candidates = [name for name, output in imported.outputs.items()
                  if output.type == "particle" and output.num_items > 0
                  and {"blob", "ctf"}.issubset({result.name for result in output.results})
                  and not any(word in name for word in ("incomplete", "rejected", "unused"))]
    if len(candidates) != 1:
        raise ValueError("Particle import must provide one non-empty particle output with blob and CTF data; check STAR image references and per-particle CTF/optics metadata")
    job = workspace.create_job("class_2D_new", connections={"particles": (imported.uid, candidates[0])},
                               title=f"Web particle Class2D {task['config']['name']}")
    task["particle_class2d_job"] = job.uid
    write(job_path(task["id"]) / "task.json", task)
    return job


def run(identifier):
    os.environ['CRYOAGENT_WEB_WORKER_PHASE'] = 'initializing'
    from webapp.usage import initialize_usage
    root = job_path(identifier)
    initialize_usage(root)
    os.environ["CRYOAGENT_USAGE_LEDGER"] = str(root / "api_usage.jsonl")
    from webapp.proxy import configure_compute_proxy
    configure_execution_policy()
    configure_compute_proxy()
    from cryosparc_client import cryosparc_client
    from webapp.gpu_routing import queue_and_wait
    task = read_task(identifier)
    config = task["config"]
    job_type, params = import_spec(config, root)
    models = task.setdefault("model_config", api_models())
    task["status"] = "running"
    task["controls_supported"] = True
    write(root / "task.json", task)
    cs = cryosparc_client()
    project = cs.find_project(os.getenv("CRYOAGENT_WEB_PROJECT", "P6"))
    if task.get("workspace"):
        workspace = project.find_workspace(task["workspace"])
    else:
        workspace = project.create_workspace(title=f"CryoSage {config['name']}", desc=f"Web task {identifier}")
        task.update(project=project.uid, workspace=workspace.uid)
        write(root / "task.json", task)
    from webapp.benchmark import configure as configure_benchmark
    configure_benchmark(task, root, cs)
    from webapp import pilot_pipeline
    pilot_pipeline.configure(task, root, cs)
    if task.get('service_finalize_checkpoint'):
        receipt = json.loads((root / 'finalization_checkpoint.json').read_text())
        finalize(task, root, receipt)
        return
    resume_args, remaining_rounds = supervised_resume(task, root)
    if remaining_rounds <= 0:
        task.update(status='needs_attention', stop_reason='decision_round_budget_reached', finished_at=now())
        write(root / 'task.json', task)
        return
    dataset = {"empiar_id": config["empiar_id"] or None, "input_type": config["input_type"],
               "pixel_size_A": config["pixel_size_A"], "accelerating_voltage_kv": config["voltage_kV"],
               "spherical_aberration_mm": config["cs_mm"], "total_exposure_dose_e_per_A2": config["dose"],
               "target_resolution_A": config["target"],
               "available_input_files": {f"{'movie' if config['input_type']=='movies' else 'micrograph'}_blob_paths": config["input_path"]}}
    dataset.update(symmetry_dataset_fields(config))
    if config.get("benchmark_profile"):
        dataset["known_workflow_steps"] = []
        dataset["benchmark_constraints"] = (
            "Treat this as a new unpublished dataset. Use only supplied raw inputs and acquisition "
            "facts, current-workspace results, and generic/non-target evidence from the frozen "
            "held-out KB. Do not seek target-specific publications, workflows, reference maps, "
            "models, known optimal parameters, or other workspaces. The actual total budget is "
            "40 created Jobs, including Import, pilots, failures and validation; never reset it."
        )
    if config["input_type"] == "particles":
        dataset["available_input_files"] = {"particle_blob_path": params["particle_blob_path"],
                                             "particle_meta_path": config["particle_star_path"]}
        dataset["particle_star_path"] = config["particle_star_path"]
        dataset["input_path"] = config["input_path"]
    write(root / "dataset.json", dataset)
    if task.get("import_job"):
        imported = workspace.find_job(task["import_job"])
    else:
        if task.get('pipeline_version') == pilot_pipeline.VERSION:
            imported = pilot_pipeline.create_preprocessing(task, root, workspace, 'import', job_type, params)
        else:
            imported = workspace.create_job(job_type, params=params, title=f"Web import {config['name']}")
        task["import_job"] = imported.uid
        write(root / "task.json", task)
    stop = threading.Event()
    monitor_thread = threading.Thread(target=monitor, args=(task, stop), daemon=True)
    monitor_thread.start()
    try:
        if queue_and_wait(imported, os.getenv("CRYOAGENT_GPU_LANE", "g8m192_4090_slurm"), 86400, 30) != "completed":
            raise RuntimeError("Dataset import failed; see CryoSPARC job errors")
        current = imported
        if pilot_pipeline.enabled(task):
            try:
                best = pilot_pipeline.run(task, root, project, workspace, imported)
            except pilot_pipeline.PipelineStopped:
                task.update(status='user_stopped', stop_reason='user_stop', finished_at=now())
                write(root/'task.json', task)
                stop.set()
                monitor_thread.join(timeout=10)
                return
            if not best:
                progress = snapshot(project.uid, workspace.uid, config['target'])
                write(root/'progress.json', progress)
                export_result(task, progress)
                task.update(status='needs_attention', stop_reason='pilot_objective_not_met', finished_at=now(),
                            error='Bounded pilot exploration did not meet the objective promotion gate; full data was not blindly promoted.')
                write(root/'task.json', task)
                stop.set()
                monitor_thread.join(timeout=10)
                return
            current = project.find_job(best)
        if config["input_type"] == "particles" and not task.get("resume_node"):
            current = particle_class2d(workspace, imported, task)
            if queue_and_wait(current, os.getenv("CRYOAGENT_GPU_LANE", "g8m192_4090_slurm"), 86400, 30) != "completed":
                raise RuntimeError("Particle Class2D failed; see CryoSPARC job errors")
    except Exception:
        stop.set()
        monitor_thread.join(timeout=10)
        raise
    command = [PYTHON, "-u", str(REPO / "cryosparc_agent/autonomous_mcp_closed_loop.py"),
               "--project", project.uid, "--workspace", workspace.uid,
               "--dataset-json-file", str(root / "dataset.json"), "--backend", "api",
               "--api-base", os.getenv("OPENAI_BASE_URL", "https://api.ofox.ai/v1"),
               "--api-model", models["planner"],
               "--fixed-api-model", models["execution"],
               "--planning-mode", "api", "--planner-max-actions", "1", "--vision", "--cryosift",
               "--kb-tool-policy", "auto" if config["kb"] else "disabled",
               "--max-rounds", str(remaining_rounds), "--max-validation-failures", "6",
               "--max-new-tokens", "8192", "--wait-timeout-seconds", "86400",
               "--poll-interval-seconds", "60", "--wait-retries", "4",
               "--output-dir", str(root / "agent")]
    command += resume_args or ['--current-node', task.get('resume_node') or current.uid]
    command += ["--user-control-dir", str(root / "control")]
    command += symmetry_runner_flags(dataset)
    prior_summaries = set((root / 'agent').glob('run_*/summary.json'))
    os.environ['CRYOAGENT_WEB_WORKER_PHASE'] = 'executing'
    log_offset = (root / 'agent.log').stat().st_size if (root / 'agent.log').exists() else 0
    with (root / "agent.log").open("a") as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=log)
        try:
            process.wait()
        finally:
            stop.set()
            monitor_thread.join(timeout=10)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    summaries = sorted(set((root / 'agent').glob('run_*/summary.json')) - prior_summaries)
    if not summaries:
        with (root / 'agent.log').open('rb') as log:
            log.seek(max(log_offset, (root / 'agent.log').stat().st_size - 8000))
            tail = log.read().decode('utf-8', errors='replace')
        raise RuntimeError('Agent exited without a current summary: ' + tail[-2000:])
    receipt = {'summary_file': str(summaries[-1]), 'runner_exit_code': process.returncode}
    write(root / 'finalization_checkpoint.json', receipt)
    finalize(task, root, receipt)


def finalize(task, root, receipt):
    """Retry exports/status only; never restart a completed scientific run."""
    os.environ['CRYOAGENT_WEB_WORKER_PHASE'] = 'finalizing'
    path = Path(receipt['summary_file']).resolve()
    if not path.is_relative_to((root / 'agent').resolve()):
        raise ValueError('Finalization summary must belong to this task')
    summary = json.loads(path.read_text())
    stop_reason = summary.get('stop_reason') or summary.get('status')
    progress = snapshot(task['project'], task['workspace'], task['config']['target'])
    write(root / "progress.json", progress)
    export_result(task, progress)
    has_map = progress.get("best_refine") is not None
    complete = receipt['runner_exit_code'] == 0 and has_map and summary.get("success") is True
    error = (summary.get("error") or {}).get("error")
    if not complete:
        task["error"] = (str(error)[:700] if error else f"Agent stopped: {stop_reason}; a complete refinement is not confirmed")
    else:
        task.pop("error", None)
    task.update(status="finished" if complete else "needs_attention", finished_at=now(),
                runner_exit_code=receipt['runner_exit_code'], stop_reason=stop_reason)
    if stop_reason == 'unattended_recovery_exhausted':
        task['status'] = 'failed'
    if stop_reason == "user_stop":
        task["status"] = "user_stopped"
        task.pop("error", None)
    task.pop('service_finalize_checkpoint', None)
    write(root / "task.json", task)


if __name__ == "__main__":
    identifier = sys.argv[1]
    try:
        run(identifier)
    except Exception as exc:
        write(job_path(identifier) / 'controller_failure.json',
              {'at': now(), 'error_type': type(exc).__name__, 'error': str(exc),
               'phase': os.getenv('CRYOAGENT_WEB_WORKER_PHASE', 'initializing')})
        task = read_task(identifier)
        task.update(status="needs_attention", error="Controller failed; inspect server-side logs", finished_at=now())
        write(job_path(identifier) / "task.json", task)
        raise
