"""Prepare per-dataset workspaces and fixed 20-micrograph subsets.

Movie inputs are sampled to 20 files, imported, and motion-corrected before
Exposure Sets. The resumable state records every workspace and CryoSPARC Job.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from cryosparc_client import cryosparc_client
from visual_rl_collection import validate_collection_config, validate_micrograph_subset


DEFAULT_SELECTION = ROOT / "data/visual_rl_30_dataset_selection_20260930.json"
DEFAULT_CANDIDATES = ROOT / "data/empiar_experimental_metadata/rl_manifest/rl_candidates.jsonl"
DEFAULT_DATA_ROOT = Path("/hdd1/linweifan/cryosparc/cryoagent_visual_rl")
BOX_PHYSICAL_A = [320 * 0.6575, 384 * 0.6575, 440 * 0.6575, 504 * 0.6575]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["empiar_id"]: row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def one_value(rows: list[dict[str, Any]], key: str, default: float | None = None) -> float:
    values = {float(item[key]) for item in rows if item.get(key) is not None}
    if len(values) > 1:
        raise ValueError(f"conflicting metadata values for {key}: {sorted(values)}")
    if values:
        return next(iter(values))
    if default is None:
        raise ValueError(f"missing required experimental metadata: {key}")
    return default


def build_dataset_record(
    dataset_id: str,
    source: dict[str, Any],
    directory_override: str | None = None,
    media_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = source["experimental_parameters"]
    voltage = one_value(info.get("accelerating_voltage_candidates", []), "value_kv", 300)
    cs = one_value(info.get("spherical_aberration_candidates", []), "value_mm", 2.7)
    dose = one_value(info.get("total_exposure_dose_candidates", []), "value_e_per_A2")

    media_type = str((media_override or {}).get("media_type", "micrographs"))
    if media_type not in {"micrographs", "movies"}:
        raise ValueError(f"unsupported media type for EMPIAR-{dataset_id}: {media_type}")
    category_token = "multiframe" if media_type == "movies" else "single frame"
    image_sets = [
        item for item in source.get("input_image_sets", [])
        if category_token in str(item.get("category", "")).lower()
        and item.get("directory")
        and item.get("pixel_width") is not None
    ]
    requested_name = (media_override or {}).get("image_set_name")
    if requested_name:
        image_sets = [item for item in image_sets if item.get("name") == requested_name]
    image_set = None
    if media_type == "movies":
        available_sets: list[tuple[int, dict[str, Any]]] = []
        for candidate in image_sets:
            rel = str(candidate["directory"]).strip("/")
            candidate_dir = Path(directory_override) if directory_override else Path("/home/share/empiar") / dataset_id / rel
            if not candidate_dir.is_dir():
                continue
            count = sum(
                1 for entry in os.scandir(candidate_dir)
                if entry.is_file() and Path(entry.name).suffix.lower() in {".mrc", ".tif", ".tiff"}
            )
            if count >= 20:
                available_sets.append((count, candidate))
        if available_sets:
            image_set = max(available_sets, key=lambda row: row[0])[1]
    else:
        for candidate in image_sets:
            rel = str(candidate["directory"]).strip("/")
            candidate_dir = Path(directory_override) if directory_override else Path("/home/share/empiar") / dataset_id / rel
            if candidate_dir.is_dir() and (not requested_name or candidate.get("name") == requested_name):
                image_set = candidate
                break
    if image_set is None:
        raise ValueError(f"no accessible {media_type} image set for EMPIAR-{dataset_id}")

    directory = (
        Path(directory_override)
        if directory_override
        else Path("/home/share/empiar") / dataset_id / str(image_set["directory"]).strip("/")
    )
    allowed_suffixes = {".mrc", ".tif", ".tiff"} if media_type == "movies" else {".mrc"}
    files = [
        Path(entry.path) for entry in os.scandir(directory)
        if entry.is_file() and Path(entry.name).suffix.lower() in allowed_suffixes
    ]
    if len(files) < 20:
        raise ValueError(f"{directory}: expected >=20 supported {media_type} files, found {len(files)}")
    suffix_counts: dict[str, int] = {}
    for path in files:
        suffix_counts[path.suffix.lower()] = suffix_counts.get(path.suffix.lower(), 0) + 1
    suffix = max(suffix_counts, key=suffix_counts.get)
    files = sorted((path for path in files if path.suffix.lower() == suffix), key=lambda path: path.name)
    if len(files) < 20:
        raise ValueError(f"{directory}: expected >=20 files with one importable suffix, found {len(files)}")
    sample = files[0]
    dims: tuple[int, ...] | None = None
    if suffix == ".mrc":
        with open(sample, "rb") as stream:
            dims = struct.unpack("<4i", stream.read(16))[:3]
        if not (0 < dims[0] < 100000 and 0 < dims[1] < 100000):
            raise ValueError(f"{sample}: invalid MRC movie dimensions {dims}")
        expected_nz = 1 if media_type == "micrographs" else image_set.get("frames_per_image")
        if expected_nz and dims[2] != int(expected_nz):
            raise ValueError(
                f"{sample}: MRC nz={dims[2]} disagrees with metadata frames_per_image={expected_nz}"
            )

    emdb_ids = source.get("emdb_ids") or []
    return {
        "dataset_id": dataset_id,
        "source": source,
        "directory": str(directory),
        "media_type": media_type,
        "image_set_name": image_set.get("name"),
        "data_format": image_set.get("data_format"),
        "frames_per_image": image_set.get("frames_per_image", 1),
        "movie_files": [str(path) for path in files],
        "movie_suffix": suffix if media_type == "movies" else None,
        "pixel_size_A": float(image_set["pixel_width"]),
        "voltage_kV": voltage,
        "cs_mm": cs,
        "dose_e_per_A2": dose,
        "parameter_defaults_used": {
            "accelerating_voltage_kV": not bool(info.get("accelerating_voltage_candidates")),
            "spherical_aberration_mm": not bool(info.get("spherical_aberration_candidates")),
        },
        "emdb_ids": emdb_ids,
        "source_file_count": len(files),
        "sample_file": sample.name,
        "sample_mrc_shape": list(dims) if dims else None,
    }


def _probe_tiff_shapes(files: list[Path]) -> dict[str, tuple[int, ...]]:
    probe_python = os.getenv("CRYOAGENT_TIFF_PROBE_PYTHON")
    if not probe_python:
        probe_python = shutil.which("python3") or shutil.which("python") or ""
    if not probe_python:
        raise RuntimeError("TIFF movie sampling requires CRYOAGENT_TIFF_PROBE_PYTHON with tifffile installed")

    probe_code = (
        "import json, sys, tifffile\n"
        "result = {}\n"
        "for name in sys.argv[1:]:\n"
        "  with tifffile.TiffFile(name) as tif:\n"
        "    page_shape = tuple(int(v) for v in tif.pages[0].shape)\n"
        "    result[name] = ([len(tif.pages)] + list(page_shape)) if len(tif.pages) > 1 else list(page_shape)\n"
        "print(json.dumps(result))\n"
    )
    result = subprocess.run(
        [probe_python, "-c", probe_code, *(str(path) for path in files)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return {name: tuple(int(value) for value in shape) for name, shape in payload.items()}


def _movie_shape(path: Path) -> tuple[int, ...]:
    if path.suffix.lower() == ".mrc":
        with open(path, "rb") as stream:
            nx, ny, nz = struct.unpack("<3i", stream.read(12))
        if not (0 < nx < 100000 and 0 < ny < 100000 and 0 < nz < 100000):
            raise ValueError(f"{path}: invalid MRC dimensions {(nx, ny, nz)}")
        return nz, ny, nx
    return ()


def prepare_movie_subset(record: dict[str, Any], target_dir: Path, seed: int) -> tuple[Path, list[str]]:
    files = [Path(path) for path in record["movie_files"]]
    rng = random.Random(seed)
    probe_order = list(files)
    rng.shuffle(probe_order)
    shape_groups: dict[tuple[int, ...], list[Path]] = {}
    for offset in range(0, len(probe_order), 32):
        chunk = probe_order[offset:offset + 32]
        tiff_files = [path for path in chunk if path.suffix.lower() in {".tif", ".tiff"}]
        tiff_shapes = _probe_tiff_shapes(tiff_files) if tiff_files else {}
        for path in chunk:
            shape = tiff_shapes[str(path)] if str(path) in tiff_shapes else _movie_shape(path)
            shape_groups.setdefault(shape, []).append(path)
        if any(len(group) >= 20 for group in shape_groups.values()):
            break
    eligible_groups = [(shape, group) for shape, group in shape_groups.items() if len(group) >= 20]
    if not eligible_groups:
        shape_counts = {"x".join(map(str, shape)): len(group) for shape, group in shape_groups.items()}
        raise ValueError(
            f"{record['directory']}: no homogeneous movie-shape group has 20 files; "
            f"shape counts={shape_counts}"
        )
    _, eligible_files = max(eligible_groups, key=lambda item: len(item[1]))
    selected = sorted(rng.sample(eligible_files, 20), key=lambda path: path.name)

    # A prior interrupted/failed attempt may have left a different random sample here.
    base_target_dir = target_dir
    conflicts = any(
        (base_target_dir / f"movie_{index:03d}{record['movie_suffix']}").is_symlink()
        and (base_target_dir / f"movie_{index:03d}{record['movie_suffix']}").resolve() != source.resolve()
        for index, source in enumerate(selected)
    )
    if conflicts:
        target_dir = base_target_dir.with_name(f"{base_target_dir.name}_shape_consistent")
    target_dir.mkdir(parents=True, exist_ok=True)
    links: list[str] = []
    for index, source_path in enumerate(selected):
        link_path = target_dir / f"movie_{index:03d}{record['movie_suffix']}"
        if link_path.is_symlink():
            if link_path.resolve() != source_path.resolve():
                raise ValueError(f"sample symlink points to an unexpected source: {link_path}")
        elif link_path.exists():
            raise FileExistsError(f"refusing to replace non-symlink sample file: {link_path}")
        else:
            link_path.symlink_to(source_path)
        links.append(str(link_path))
    return target_dir, links


def scaled_box_sizes(pixel_size_A: float) -> list[int]:
    choices = sorted({max(32, min(720, int(round((width / pixel_size_A) / 8) * 8))) for width in BOX_PHYSICAL_A})
    return choices


def build_collection_config(batch_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: dict[str, str] = {}
    dataset_info: dict[str, Any] = {}
    dataset_parameter_space: dict[str, Any] = {}
    for record in records:
        dataset_id = record["dataset_id"]
        source = record["source"]
        key = f"EMPIAR-{dataset_id}"
        datasets[key] = "train"
        dataset_info[key] = {
            "empiar_id": key,
            "emdb_id": ", ".join(record["emdb_ids"]),
            "input_type": "micrographs",
            "macromolecules_type": source.get("title"),
            "pixel_size_A": record["pixel_size_A"],
            "accelerating_voltage_kV": record["voltage_kV"],
            "spherical_aberration_mm": record["cs_mm"],
            "total_exposure_dose_e_per_A2": record["dose_e_per_A2"],
            "micrograph_path": record["directory"],
            "micrograph_format": (
                f"{record['data_format']} movies, motion-corrected before Patch CTF"
                if record["media_type"] == "movies"
                else "MRC single-frame micrographs"
            ),
            "source_image_set": record["image_set_name"],
            "source_media_type": record["media_type"],
            "parameter_sources": "EMPIAR image-set metadata and EMDB experimental metadata; approved defaults are recorded per dataset in the preparation state.",
        }
        dataset_parameter_space[key] = {
            "extract_box_size": {"box_size_pix": scaled_box_sizes(record["pixel_size_A"])},
        }

    config = {
        "experiment_id": batch_id,
        "datasets": datasets,
        "dataset_info": dataset_info,
        "parameter_space": {
            "blob_picker_diameter": {"diameter": [80, 240], "diameter_max": [140, 320]},
            "inspect_picks_threshold": {
                "ncc_score_min": [0, 1],
                "local_power_min": [0, 5000],
                "local_power_max": [0, 8000],
            },
            "extract_box_size": {"box_size_pix": [320, 384, 440, 504]},
        },
        "dataset_parameter_space": dataset_parameter_space,
        "limits": {
            "max_trials_per_dataset": 1,
            "max_trials_total": len(records),
            "max_jobs_per_trial": 4,
            "max_decisions_per_stage": 2,
        },
        "sampling": {
            "method": "cryosparc_exposure_sets_tool",
            "micrograph_count": 20,
            "randomize": True,
            "reuse_subset_across_trials": True,
        },
        "execution": {"class2D_parameters": {"class2D_K": 50, "compute_num_gpus": 1}},
        "notes": [
            "Thirty-dataset high-throughput picker/2D feedback survey; not a full reconstruction benchmark.",
            "Each trial uses its own deterministic 20-micrograph Exposure Sets split; movie inputs are first sampled to 20 files and Patch Motion Corrected.",
            "Extract box-size candidates preserve approximately the pilot's physical box-width grid, scaled by each image-set pixel size and rounded to 8 pixels.",
            "Projection-consistency feedback requires a verified reference-map path; no reference maps are inferred here.",
        ],
    }
    return validate_collection_config(config)


def queue_and_wait(job: Any, lane: str, timeout_s: int, poll_s: int) -> str:
    status = str(job.status).lower()
    if status == "building":
        job.queue(lane=lane)
        status = str(job.status).lower()
        print(f"queued {job.uid} status={status}", flush=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job.refresh()
        status = str(job.status).lower()
        if status in {"completed", "failed", "killed"}:
            return status
        time.sleep(poll_s)
    raise TimeoutError(f"job {job.uid} did not finish within {timeout_s}s; last status={status}")


def workspace_for(project: Any, title: str, saved_uid: str | None) -> Any:
    if saved_uid:
        return project.find_workspace(saved_uid)
    for workspace in project.find_workspaces():
        if workspace.title == title:
            return workspace
    return project.create_workspace(title=title, desc="Visual RL fixed-20-micrograph decision-trace collection.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    selection = read_json(Path(args.selection))
    candidates = load_candidates(Path(args.candidates))
    dataset_ids = [str(item) for item in selection["datasets"]]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("selection contains duplicate EMPIAR IDs")
    directory_overrides = selection.get("input_directory_overrides", {})
    media_overrides = selection.get("input_media_overrides", {})
    for dataset_id in selection.get("movie_dataset_ids", []):
        media_overrides.setdefault(str(dataset_id), {"media_type": "movies"})
    records = [
        build_dataset_record(
            dataset_id,
            candidates[dataset_id],
            str(directory_overrides[dataset_id]) if dataset_id in directory_overrides else None,
            media_overrides.get(dataset_id),
        )
        for dataset_id in dataset_ids
    ]

    batch_id = args.batch_id or selection["batch_id"]
    output_dir = Path(args.data_root) / "runs" / "preparation" / batch_id
    state_path = output_dir / "preparation_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "batch_id": batch_id,
        "project_uid": args.project,
        "status": "planned",
        "datasets": {r["dataset_id"]: {"status": "pending"} for r in records},
        "created_at": utc_now(),
    }
    if state["project_uid"] != args.project:
        raise ValueError("saved preparation state belongs to another CryoSPARC project")
    config = build_collection_config(batch_id, records)

    if not args.execute:
        return {
            "status": "planned",
            "batch_id": batch_id,
            "project_uid": args.project,
            "dataset_count": len(records),
            "sampled_mrc_paths": [r["directory"] for r in records],
            "collection_config_valid": True,
            "collection_config_preview": config,
            "state_path": str(state_path),
            "note": "No CryoSPARC objects or jobs were created. Pass --execute to prepare workspaces and jobs.",
        }

    cs = cryosparc_client()
    project = cs.find_project(args.project)
    state["status"] = "preparing"
    state["started_at"] = state.get("started_at") or utc_now()
    write_json(state_path, state)

    for index, record in enumerate(records):
        if args.limit is not None and index >= args.limit:
            break
        dataset_id = record["dataset_id"]
        item = state["datasets"].setdefault(dataset_id, {"status": "pending"})
        if item.get("status") == "ready":
            print(
                f"reuse ready EMPIAR-{dataset_id}: workspace={item.get('workspace_uid')} "
                f"exposure_sets={item.get('exposure_sets_job_uid')}",
                flush=True,
            )
            continue
        workspace_title = f"{batch_id} - EMPIAR-{dataset_id}"
        workspace = workspace_for(project, workspace_title, item.get("workspace_uid"))
        item.update({"workspace_uid": workspace.uid, "workspace_title": workspace.title})
        retry_index = len(item.get("previous_attempts", []))
        prior_import_uid = item.get("import_job_uid")
        if prior_import_uid and record["media_type"] == "movies":
            prior_import_job = workspace.find_job(prior_import_uid)
            prior_import_status = str(prior_import_job.status).lower()
            if prior_import_status in {"failed", "killed"}:
                item.setdefault("previous_attempts", []).append({
                    "import_job_uid": prior_import_uid,
                    "import_status": prior_import_status,
                    "sampled_movie_paths": item.get("metadata", {}).get("sampled_movie_paths", []),
                    "reason": "CryoSPARC Import Movies failed while decoding a selected movie",
                })
                for key in (
                    "import_job_uid", "import_status", "motion_correction_job_uid",
                    "motion_correction_status", "exposure_sets_job_uid", "exposure_sets_status",
                ):
                    item.pop(key, None)
                item["status"] = "retrying_import_with_new_movie_sample"
                retry_index = len(item["previous_attempts"])
                write_json(state_path, state)
        item["metadata"] = {k: record[k] for k in (
            "directory", "media_type", "image_set_name", "data_format", "frames_per_image",
            "pixel_size_A", "voltage_kV", "cs_mm", "dose_e_per_A2", "emdb_ids",
            "parameter_defaults_used", "source_file_count", "sample_file", "sample_mrc_shape",
        )}
        write_json(state_path, state)

        import_dir = Path(record["directory"])
        if record["media_type"] == "movies":
            import_dir, selected_movie_paths = prepare_movie_subset(
                record,
                output_dir / "sampled_movies" / (
                    f"empiar_{dataset_id}_retry_{retry_index:02d}" if retry_index else f"empiar_{dataset_id}"
                ),
                seed=(sum(ord(char) for char in f"{batch_id}:{dataset_id}:{retry_index}")),
            )
            item["metadata"]["source_movie_count"] = len(record["movie_files"])
            item["metadata"]["sampled_movie_count"] = len(selected_movie_paths)
            item["metadata"]["sampled_movie_paths"] = selected_movie_paths
            item["metadata"]["import_movie_glob"] = str(import_dir / f"*{record['movie_suffix']}")
            write_json(state_path, state)

        import_uid = item.get("import_job_uid")
        if import_uid:
            import_job = workspace.find_job(import_uid)
            if record["media_type"] == "movies" and str(import_job.status).lower() == "completed":
                imported_count = len(list(import_job.load_output("imported_movies").rows()))
                if imported_count != 20:
                    item.setdefault("previous_attempts", []).append({
                        "import_job_uid": import_uid,
                        "imported_movie_count": imported_count,
                        "motion_correction_job_uid": item.get("motion_correction_job_uid"),
                        "exposure_sets_job_uid": item.get("exposure_sets_job_uid"),
                        "reason": "import did not produce the required 20 homogeneous movies",
                    })
                    for key in (
                        "import_job_uid", "import_status", "motion_correction_job_uid",
                        "motion_correction_status", "exposure_sets_job_uid", "exposure_sets_status",
                    ):
                        item.pop(key, None)
                    item["status"] = "retrying_with_shape_consistent_sample"
                    write_json(state_path, state)
                    import_uid = None
        else:
            import_job = None
        if not import_uid:
            import_job_type = "import_movies" if record["media_type"] == "movies" else "import_micrographs"
            import_glob = (
                str(import_dir / f"*{record['movie_suffix']}")
                if record["media_type"] == "movies"
                else str(import_dir / "*.mrc")
            )
            import_desc = (
                "Import a deterministic 20-movie sample for motion correction and visual RL collection."
                if record["media_type"] == "movies"
                else "Import validated 2D MRC micrographs for a fixed-20-micrograph visual RL trial."
            )
            import_params = {
                "blob_paths": import_glob,
                "psize_A": record["pixel_size_A"],
                "accel_kv": record["voltage_kV"],
                "cs_mm": record["cs_mm"],
                "total_dose_e_per_A2": record["dose_e_per_A2"],
                # A 20-movie sample is small enough to validate frame consistency.
                "skip_header_check": record["media_type"] != "movies",
            }
            import_job = workspace.create_job(
                import_job_type,
                connections={},
                params=import_params,
                title=f"RL import {'movies' if record['media_type'] == 'movies' else 'micrographs'} EMPIAR-{dataset_id}",
                desc=import_desc,
            )
            item["import_job_uid"] = import_job.uid
            item["status"] = "import_created"
            write_json(state_path, state)
        if str(import_job.status).lower() not in {"completed"}:
            status = queue_and_wait(
                import_job, args.cpu_lane, args.job_timeout_seconds, args.poll_interval_seconds,
            )
            item["import_status"] = status
            if status != "completed":
                item["status"] = "import_failed"
                write_json(state_path, state)
                raise RuntimeError(f"EMPIAR-{dataset_id} import job {import_job.uid} ended as {status}")
        item["status"] = "import_completed"
        write_json(state_path, state)

        exposure_source_uid = import_job.uid
        exposure_source_output = "imported_micrographs"
        if record["media_type"] == "movies":
            motion_uid = item.get("motion_correction_job_uid")
            if motion_uid:
                motion_job = workspace.find_job(motion_uid)
            else:
                motion_job = workspace.create_job(
                    "patch_motion_correction_multi",
                    connections={"movies": (import_job.uid, "imported_movies")},
                    params={"compute_num_gpus": 1},
                    title=f"RL Patch Motion Correction EMPIAR-{dataset_id}",
                    desc="Motion-correct the fixed 20-movie RL sample before CTF estimation.",
                )
                item["motion_correction_job_uid"] = motion_job.uid
                item["status"] = "motion_correction_created"
                write_json(state_path, state)
            if str(motion_job.status).lower() not in {"completed"}:
                status = queue_and_wait(
                    motion_job, args.gpu_lane, args.job_timeout_seconds, args.poll_interval_seconds,
                )
                item["motion_correction_status"] = status
                if status != "completed":
                    item["status"] = "motion_correction_failed"
                    write_json(state_path, state)
                    raise RuntimeError(
                        f"EMPIAR-{dataset_id} Patch Motion Correction {motion_job.uid} ended as {status}"
                    )
            exposure_source_uid = motion_job.uid
            exposure_source_output = "micrographs"
            item["motion_correction_status"] = "completed"
            write_json(state_path, state)

        exposure_uid = item.get("exposure_sets_job_uid")
        if exposure_uid:
            exposure_job = workspace.find_job(exposure_uid)
        else:
            exposure_job = workspace.create_job(
                "exposure_sets",
                connections={"exposures_A": (exposure_source_uid, exposure_source_output)},
                params={"set_operation": "split", "set_split_num": 2, "set_split_size": 20, "set_split_random": True},
                title=f"RL randomized 20 micrographs EMPIAR-{dataset_id}",
                desc="Randomized fixed 20-micrograph subset for visual RL decision-trace collection.",
            )
            item["exposure_sets_job_uid"] = exposure_job.uid
            item["status"] = "exposure_sets_created"
            write_json(state_path, state)
        if str(exposure_job.status).lower() not in {"completed"}:
            status = queue_and_wait(
                exposure_job, args.cpu_lane, args.job_timeout_seconds, args.poll_interval_seconds,
            )
            item["exposure_sets_status"] = status
            if status != "completed":
                item["status"] = "exposure_sets_failed"
                write_json(state_path, state)
                raise RuntimeError(f"EMPIAR-{dataset_id} Exposure Sets job {exposure_job.uid} ended as {status}")

        subset = exposure_job.load_output("split_0")
        subset_payload = {
            "source_job_uid": exposure_job.uid,
            "output_name": "split_0",
            "micrograph_uids": [str(row["uid"]) for row in subset.rows()],
        }
        checked = validate_micrograph_subset(subset_payload)
        item.update({
            "status": "ready",
            "exposure_sets_status": "completed",
            "subset_fingerprint": checked["fingerprint"],
            "subset_count": checked["micrograph_count"],
        })
        write_json(output_dir / "subsets" / f"empiar_{dataset_id}.json", checked)
        write_json(state_path, state)
        print(
            f"ready EMPIAR-{dataset_id}: workspace={workspace.uid} import={import_job.uid} "
            f"motion={item.get('motion_correction_job_uid', 'not-needed')} "
            f"exposure_sets={exposure_job.uid} n={checked['micrograph_count']}",
            flush=True,
        )

    ready_records = [r for r in records if state["datasets"].get(r["dataset_id"], {}).get("status") == "ready"]
    config = build_collection_config(batch_id, ready_records)
    manifest = {
        "batch_id": batch_id,
        "trials": [
            {
                "dataset_id": f"EMPIAR-{r['dataset_id']}",
                "project": args.project,
                "workspace": state["datasets"][r["dataset_id"]]["workspace_uid"],
                "subset_job_uid": state["datasets"][r["dataset_id"]]["exposure_sets_job_uid"],
            }
            for r in ready_records
        ],
    }
    state["status"] = "ready" if len(ready_records) == len(records) else "partial"
    state["ready_dataset_count"] = len(ready_records)
    state["updated_at"] = utc_now()
    write_json(state_path, state)
    write_json(output_dir / "collection_config.json", config)
    write_json(output_dir / "batch_manifest.json", manifest)
    return {
        "status": state["status"],
        "batch_id": batch_id,
        "ready_dataset_count": len(ready_records),
        "total_dataset_count": len(records),
        "config_path": str(output_dir / "collection_config.json"),
        "manifest_path": str(output_dir / "batch_manifest.json"),
        "state_path": str(state_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--batch-id")
    parser.add_argument("--project", default="P19")
    parser.add_argument("--cpu-lane", default="g8m192_4090_slurm",
                        help="CryoSPARC lane for CPU-only preparation jobs")
    parser.add_argument("--gpu-lane", default="g8m192_4090_slurm",
                        help="CryoSPARC lane for movie motion correction")
    parser.add_argument("--limit", type=int, help="Only prepare the first N datasets")
    parser.add_argument("--execute", action="store_true", help="Create/queue CryoSPARC import and Exposure Sets jobs")
    parser.add_argument("--job-timeout-seconds", type=int, default=43200)
    parser.add_argument("--poll-interval-seconds", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
