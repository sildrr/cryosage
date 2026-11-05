"""Run Patch CTF for every prepared 20-micrograph subset before RL collection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from cryosparc_client import cryosparc_client
from visual_rl_collection import (
    FIXED_MICROGRAPH_COUNT,
    normalized_dataset_id,
    read_json,
    write_json,
)


DEFAULT_BATCH_DIR = Path(
    "/hdd1/linweifan/cryosparc/cryoagent_visual_rl/runs/preparation/visual-rl-30-datasets-20260930"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def output_count(job: Any) -> int:
    return len(output_uids(job))


def output_uids(job: Any) -> list[str]:
    return [str(row["uid"]) for row in job.load_output("exposures").rows()]


def validate_ctf_output(job: Any, source_uids: list[str]) -> dict[str, Any]:
    output_uids_list = output_uids(job)
    output_set = set(output_uids_list)
    source_set = {str(uid) for uid in source_uids}
    if not output_set or len(output_uids_list) != len(output_set):
        raise ValueError(f"Patch CTF {job.uid} has no exposures or duplicate UIDs")
    unexpected = output_set - source_set
    if unexpected:
        raise ValueError(f"Patch CTF {job.uid} returned exposures outside the bound subset: {sorted(unexpected)}")
    return {
        "output_count": len(output_set),
        "retained_micrograph_uids": sorted(output_set),
        "dropped_micrograph_uids": sorted(source_set - output_set),
    }


def wait_for_job(job: Any, timeout_s: int, poll_s: int) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job.refresh()
        status = str(job.status).lower()
        if status in {"completed", "failed", "killed"}:
            return status
        time.sleep(poll_s)
    raise TimeoutError(f"Patch CTF {job.uid} did not finish within {timeout_s}s; last status={job.status}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    batch_dir = Path(args.batch_dir)
    prep_state_path = batch_dir / "preparation_state.json"
    manifest_path = batch_dir / "batch_manifest.json"
    prep_state = read_json(prep_state_path)
    manifest = read_json(manifest_path)
    rows = manifest.get("trials") or []
    if prep_state.get("status") != "ready" or len(rows) != args.expected_datasets:
        raise ValueError(
            f"expected a ready {args.expected_datasets}-dataset preparation; "
            f"state={prep_state.get('status')}, manifest_rows={len(rows)}"
        )

    state_path = batch_dir / "ctf_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "batch_id": manifest["batch_id"],
        "project_uid": args.project,
        "status": "planned",
        "datasets": {},
        "created_at": utc_now(),
    }
    if state.get("project_uid") != args.project or state.get("batch_id") != manifest["batch_id"]:
        raise ValueError("saved CTF state belongs to a different project or batch")

    cs = cryosparc_client()
    project = cs.find_project(args.project)
    state["status"] = "running"
    state["started_at"] = state.get("started_at") or utc_now()
    write_json(state_path, state)

    for index, row in enumerate(rows):
        dataset_id = str(row["dataset_id"])
        dataset_key = normalized_dataset_id(dataset_id)
        entry = state["datasets"].get(dataset_key)
        if entry is None:
            entry = state["datasets"].pop(dataset_id, {"status": "pending"})
            state["datasets"][dataset_key] = entry
        if entry.get("status") == "completed":
            job = project.find_job(entry["ctf_job_uid"])
            subset = read_json(batch_dir / "subsets" / f"empiar_{dataset_key}.json")
            if str(job.status).lower() == "completed":
                details = validate_ctf_output(job, subset["micrograph_uids"])
                row["ctf_job_uid"] = job.uid
                row["ctf_output_count"] = details["output_count"]
                row["ctf_retained_micrograph_uids"] = details["retained_micrograph_uids"]
                row["ctf_dropped_micrograph_uids"] = details["dropped_micrograph_uids"]
                print(
                    f"verified EMPIAR-{dataset_key}: Patch CTF {job.uid} "
                    f"retained={details['output_count']}/{FIXED_MICROGRAPH_COUNT}",
                    flush=True,
                )
                continue
            entry["status"] = "needs_recovery"

        workspace = project.find_workspace(str(row["workspace"]))
        job_uid = entry.get("ctf_job_uid")
        if job_uid:
            job = workspace.find_job(job_uid)
        else:
            source_uid = str(row["subset_job_uid"])
            title = f"RL Patch CTF EMPIAR-{dataset_key}"
            job = next(
                (
                    existing for existing in workspace.find_jobs()
                    if existing.title == title and existing.type == "patch_ctf_estimation_multi"
                ),
                None,
            )
            if job is None:
                job = workspace.create_job(
                    "patch_ctf_estimation_multi",
                    connections={"exposures": (source_uid, "split_0")},
                    params={"compute_num_gpus": 1},
                    title=title,
                    desc="Estimate CTF on the fixed randomized 20-micrograph visual RL subset before Blob Picker.",
                )
            entry.update({
                "ctf_job_uid": job.uid,
                "source_job_uid": source_uid,
                "source_output": "split_0",
                "status": "created",
                "created_at": utc_now(),
            })
            write_json(state_path, state)

        status = str(job.status).lower()
        if status == "building":
            job.queue(lane=args.gpu_lane)
            entry.update({"status": "queued", "lane": args.gpu_lane, "queued_at": utc_now()})
            write_json(state_path, state)
            print(f"queued EMPIAR-{dataset_key}: {job.uid} lane={args.gpu_lane}", flush=True)
        elif status not in {"queued", "launched", "started", "running", "waiting", "completed"}:
            raise RuntimeError(f"Patch CTF {job.uid} has unexpected state {status}")

        status = wait_for_job(job, args.job_timeout_seconds, args.poll_interval_seconds)
        entry["status"] = status
        entry["finished_at"] = utc_now()
        write_json(state_path, state)
        if status != "completed":
            state["status"] = "failed"
            state["failed_dataset_id"] = dataset_key
            write_json(state_path, state)
            raise RuntimeError(f"EMPIAR-{dataset_key} Patch CTF {job.uid} ended as {status}")

        subset = read_json(batch_dir / "subsets" / f"empiar_{dataset_key}.json")
        try:
            details = validate_ctf_output(job, subset["micrograph_uids"])
        except ValueError as exc:
            state["status"] = "failed"
            entry["status"] = "invalid_output"
            entry["error"] = str(exc)
            write_json(state_path, state)
            raise
        count = details["output_count"]
        entry.update({"status": "completed", **details})
        row["ctf_job_uid"] = job.uid
        row["ctf_output_count"] = count
        row["ctf_retained_micrograph_uids"] = details["retained_micrograph_uids"]
        row["ctf_dropped_micrograph_uids"] = details["dropped_micrograph_uids"]
        prep_state["datasets"][dataset_key]["patch_ctf_job_uid"] = job.uid
        prep_state["datasets"][dataset_key]["patch_ctf_status"] = "completed"
        prep_state["datasets"][dataset_key]["patch_ctf_output_count"] = count
        prep_state["datasets"][dataset_key]["patch_ctf_retained_micrograph_uids"] = details["retained_micrograph_uids"]
        prep_state["datasets"][dataset_key]["patch_ctf_dropped_micrograph_uids"] = details["dropped_micrograph_uids"]
        write_json(prep_state_path, prep_state)
        write_json(state_path, state)
        print(
            f"completed EMPIAR-{dataset_key}: Patch CTF {job.uid}, "
            f"retained={count}/{FIXED_MICROGRAPH_COUNT}, dropped={len(details['dropped_micrograph_uids'])}",
            flush=True,
        )

    if not all(row.get("ctf_job_uid") for row in rows):
        raise RuntimeError("not every dataset has a completed Patch CTF job")
    state["status"] = "completed"
    state["completed_dataset_count"] = len(rows)
    state["updated_at"] = utc_now()
    prep_state["ctf_status"] = "completed"
    prep_state["ctf_completed_dataset_count"] = len(rows)
    prep_state["updated_at"] = utc_now()
    write_json(state_path, state)
    write_json(prep_state_path, prep_state)
    write_json(manifest_path, manifest)
    write_json(batch_dir / "ctf_manifest.json", {
        "batch_id": manifest["batch_id"],
        "project": args.project,
        "completed_dataset_count": len(rows),
        "trials": rows,
        "completed_at": utc_now(),
    })
    return {
        "status": "completed",
        "batch_id": manifest["batch_id"],
        "completed_dataset_count": len(rows),
        "state_path": str(state_path),
        "collection_manifest": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    parser.add_argument("--project", default="P19")
    parser.add_argument("--gpu-lane", default="g8m192_4090_slurm")
    parser.add_argument("--expected-datasets", type=int, default=30)
    parser.add_argument("--job-timeout-seconds", type=int, default=43200)
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
