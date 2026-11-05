"""Merge the three prepared batches and repair the power-spectrum-contaminated subset."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

from cryosparc_client import cryosparc_client
from prepare_visual_rl_batch import queue_and_wait
from prepare_visual_rl_ctf_batch import validate_ctf_output
from visual_rl_collection import validate_collection_config, validate_micrograph_subset, write_json


DATA_ROOT = Path("/hdd1/linweifan/cryosparc/cryoagent_visual_rl")
BATCH_ID = "visual-rl-100-collection-20261009"
BATCHES = (
    "visual-rl-30-datasets-20260930",
    "visual-rl-100-datasets-20261001",
    "visual-rl-movie-54-20261008",
)


def read(path: Path):
    return json.loads(path.read_text())


def repair_subset(output_dir: Path, metadata: dict, workspace_uid: str) -> dict:
    checkpoint = output_dir / "repair_empiar_12486.json"
    state = read(checkpoint) if checkpoint.exists() else {"dataset_id": "EMPIAR-12486"}
    sample_dir = output_dir / "sampled_micrographs" / "empiar_12486"
    sample_dir.mkdir(parents=True, exist_ok=True)
    # _PS.mrc files are power spectra saved alongside the real micrographs.
    files = sorted(
        Path(entry.path) for entry in os.scandir(metadata["directory"])
        if entry.is_file() and entry.name.lower().endswith("_fractions.mrc")
    )
    if len(files) < 20:
        raise ValueError(f"EMPIAR-12486 has only {len(files)} real micrograph candidates")
    selected = sorted(random.Random(20261009).sample(files, 20))
    for index, source in enumerate(selected):
        link = sample_dir / f"micrograph_{index:03d}.mrc"
        if link.is_symlink():
            if link.resolve() != source.resolve():
                raise ValueError(f"unexpected existing sample link: {link}")
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(source)
    state["sampled_source_paths"] = [str(path) for path in selected]
    state["excluded_file_pattern"] = "*_PS.mrc"
    workspace = cryosparc_client().find_project("P19").find_workspace(workspace_uid)

    def stage(key, job_type, connections, params):
        if state.get(key):
            job = workspace.find_job(state[key])
        else:
            job = workspace.create_job(
                job_type, connections=connections, params=params,
                title=f"RL100 corrected EMPIAR-12486 {job_type}",
                desc="Fixed 20 real micrographs; power-spectrum files excluded.",
            )
            state[key] = job.uid
            write_json(checkpoint, state)
        status = queue_and_wait(job, "g8m192_4090_slurm", 43200, 20)
        state[f"{key}_status"] = status
        write_json(checkpoint, state)
        if status != "completed":
            raise RuntimeError(f"EMPIAR-12486 repair {job_type} {job.uid} ended as {status}")
        return job

    imported = stage("import_job_uid", "import_micrographs", {}, {
        "blob_paths": str(sample_dir / "*.mrc"), "psize_A": metadata["pixel_size_A"],
        "accel_kv": metadata["voltage_kV"], "cs_mm": metadata["cs_mm"],
        "total_dose_e_per_A2": metadata["dose_e_per_A2"], "skip_header_check": False,
    })
    exposure = stage("subset_job_uid", "exposure_sets", {
        "exposures_A": (imported.uid, "imported_micrographs"),
    }, {"set_operation": "split", "set_split_num": 2, "set_split_size": 20, "set_split_random": True})
    subset = validate_micrograph_subset({
        "source_job_uid": exposure.uid, "output_name": "split_0",
        "micrograph_uids": [str(row["uid"]) for row in exposure.load_output("split_0").rows()],
    })
    write_json(output_dir / "subsets" / "empiar_12486.json", subset)
    ctf = stage("ctf_job_uid", "patch_ctf_estimation_multi", {
        "exposures": (exposure.uid, "split_0"),
    }, {"compute_num_gpus": 1})
    details = validate_ctf_output(ctf, subset["micrograph_uids"])
    state.update({"status": "completed", **details})
    write_json(checkpoint, state)
    print(f"repaired EMPIAR-12486: import={imported.uid} subset={exposure.uid} CTF={ctf.uid}", flush=True)
    return {
        "dataset_id": "EMPIAR-12486", "project": "P19", "workspace": workspace_uid,
        "subset_job_uid": exposure.uid, "ctf_job_uid": ctf.uid,
        "ctf_output_count": details["output_count"],
        "ctf_retained_micrograph_uids": details["retained_micrograph_uids"],
        "ctf_dropped_micrograph_uids": details["dropped_micrograph_uids"],
    }


def run(execute: bool):
    output_dir = DATA_ROOT / "runs" / "preparation" / BATCH_ID
    rows, configs, seen = [], [], set()
    repair_metadata, repair_workspace = None, None
    for batch in BATCHES:
        source_dir = DATA_ROOT / "runs" / "preparation" / batch
        configs.append(read(source_dir / "collection_config.json"))
        ctf_state = read(source_dir / "ctf_state.json")
        for original in read(source_dir / "batch_manifest.json")["trials"]:
            row = copy.deepcopy(original)
            dataset_id = row["dataset_id"]
            if dataset_id in seen:
                raise ValueError(f"duplicate dataset: {dataset_id}")
            seen.add(dataset_id)
            key = dataset_id.removeprefix("EMPIAR-")
            entry = ctf_state["datasets"][key]
            if entry["status"] != "completed":
                if key != "12486":
                    raise ValueError(f"unexpected unready CTF: {dataset_id}")
                prep = read(source_dir / "preparation_state.json")
                repair_metadata = prep["datasets"][key]["metadata"]
                repair_workspace = row["workspace"]
            else:
                row.update({
                    "ctf_job_uid": entry["ctf_job_uid"],
                    "ctf_output_count": entry["output_count"],
                    "ctf_retained_micrograph_uids": entry["retained_micrograph_uids"],
                    "ctf_dropped_micrograph_uids": entry["dropped_micrograph_uids"],
                })
            rows.append(row)
    if len(seen) != 100:
        raise ValueError(f"expected 100 distinct datasets, found {len(seen)}")
    config = copy.deepcopy(configs[0])
    config["experiment_id"] = BATCH_ID
    for field in ("datasets", "dataset_info", "dataset_parameter_space"):
        config[field] = {}
        for source in configs:
            config[field].update(source[field])
    config["limits"]["max_trials_total"] = 100
    config["notes"] = [
        "One fresh local-Qwen trajectory per dataset, fixed 20-micrograph subsets, 100 distinct datasets.",
        "Three model decisions: Blob Picker diameter, Inspect Picks thresholds, Extract box size; then Class 2D quality feedback.",
        "EMPIAR-12486 uses a repaired subset that excludes ancillary power-spectrum files.",
        "Original batch artifacts are preserved; unavailable metrics and failed trajectories remain explicitly recorded.",
    ]
    validate_collection_config(config)
    if not execute:
        return {"status": "planned", "dataset_count": len(rows), "repair_dataset": "EMPIAR-12486", "output_dir": str(output_dir)}
    repaired = repair_subset(output_dir, repair_metadata, repair_workspace)
    rows = [repaired if row["dataset_id"] == "EMPIAR-12486" else row for row in rows]
    write_json(output_dir / "collection_config.json", config)
    write_json(output_dir / "batch_manifest.json", {"batch_id": BATCH_ID, "source_batches": list(BATCHES), "trials": rows})
    result = {"status": "ready", "batch_id": BATCH_ID, "dataset_count": len(rows), "ctf_ready_count": sum(bool(row.get("ctf_job_uid")) for row in rows)}
    write_json(output_dir / "merge_state.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.execute), indent=2), flush=True)
