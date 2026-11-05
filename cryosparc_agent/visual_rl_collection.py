"""Bounded visual-parameter self-play decisions and outcome ledger.

It proposes one stage decision at a time, validates it against an experiment
search space, and records decisions/outcomes. Job submission is only possible
through the existing MCP validator/executor and requires an explicit --submit.
"""

from __future__ import annotations

import argparse
import atexit
import asyncio
import base64
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_direct_runner import parse_model_decision_text, resolve_api_key, run_openai_compatible_model
from visual_decision_contract import STAGE_ACTIONS, STAGE_PARAMETER_FIELDS, build_stage_decision_messages


SCHEMA_VERSION = "cryoagent-visual-rl-collection-0.3"
STAGES = ("blob_picker_diameter", "inspect_picks_threshold", "extract_box_size")
EVENT_STAGES = (*STAGES, "class_2d", "class2d_feedback", "execution")
FIXED_MICROGRAPH_COUNT = 20
DEFAULT_DATA_ROOT = Path(
    os.getenv(
        "CRYOAGENT_RL_DATA_ROOT",
        "/hdd1/linweifan/cryosparc/cryoagent_visual_rl",
    )
).expanduser()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_attempt_events(trial):
    events = trial["events"]
    for index in range(len(events) - 1, -1, -1):
        if events[index]["event_type"] == "retry_started":
            return events[index + 1:]
    return events


def exception_leaves(exc):
    children = getattr(exc, "exceptions", None)
    if children:
        return [leaf for child in children for leaf in exception_leaves(child)]
    return [{"type": type(exc).__name__, "message": str(exc)}]


class CollectionServiceUnavailable(RuntimeError):
    """Service retry deadline reached; preserve the pending episode, not a reward."""


def is_transient_service_error(exc):
    for leaf in exception_leaves(exc):
        message = leaf["message"].lower()
        if "model worker" in message or "cuda" in message:
            continue
        if re.search(r"(?:code\s+|http(?:/\d(?:\.\d)?)?\s+|status(?:_code)?[\s:=]+)(?:5\d\d|429)\b", message):
            return True
        if leaf["type"] in {"ConnectError", "ConnectTimeout", "ReadTimeout", "ConnectionError"}:
            return True
        if any(text in message for text in (
            "connection refused", "connection reset", "remote protocol error",
            "server disconnected", "temporary failure in name resolution",
        )):
            return True
    return False


async def run_trial_with_service_retries(args, state, item, state_path):
    start = None
    delay = max(0.1, float(getattr(args, "service_retry_delay_seconds", 30)))
    max_delay = max(delay, float(getattr(args, "service_retry_max_delay_seconds", 300)))
    max_wait = max(0.0, float(getattr(args, "service_retry_max_wait_seconds", 21600)))
    retry_count = 0
    while True:
        try:
            result = await run_trial(args)
            state.pop("service_wait", None)
            state["status"] = "running"
            item["status"] = "running"
            return result
        except Exception as exc:
            if not is_transient_service_error(exc):
                state.pop("service_wait", None)
                state["status"] = "running"
                raise
            if start is None:
                start = time.monotonic()
            remaining = max_wait - (time.monotonic() - start)
            retry_count += 1
            wait = min(delay, max(0.0, remaining))
            status = "waiting_service" if remaining > 0 else "service_unavailable"
            item["status"] = "waiting_service"
            state["status"] = status
            state["service_wait"] = {
                "trial_id": item["trial_id"], "dataset_id": item["dataset_id"],
                "retry_count": retry_count, "retry_delay_seconds": wait,
                "leaf_errors": exception_leaves(exc), "updated_at": utc_now(),
            }
            state["updated_at"] = utc_now()
            write_json(state_path, state)
            print(json.dumps({"event": status, **state["service_wait"]}), flush=True)
            if remaining <= 0:
                raise CollectionServiceUnavailable(
                    f"CryoSPARC unavailable for this retry window; resume {state_path}"
                ) from exc
            await asyncio.sleep(wait)
            delay = min(delay * 2, max_delay)


def normalized_dataset_id(value: str) -> str:
    value = str(value).lower()
    if value.startswith("empiar-"):
        value = value[len("empiar-") :]
    return value[:-1] if value.endswith("m") and value[:-1].isdigit() else value


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def snapshot_exposure_subset(args: argparse.Namespace) -> dict[str, Any]:
    """Read actual exposure UIDs from an Exposure Sets Tool output, without mutation."""
    from cryosparc_client import cryosparc_client

    cs = cryosparc_client()
    project = cs.find_project(args.project)
    job = project.find_job(args.job_uid)
    dataset = job.load_output(args.output_name)
    subset = {
        "source_job_uid": args.job_uid,
        "output_name": args.output_name,
        "micrograph_uids": [str(row["uid"]) for row in dataset.rows()],
    }
    checked = validate_micrograph_subset(subset)
    write_json(args.output, checked)
    return {"output_file": args.output, **checked}


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def ensure_writable_data_root(path: str | Path) -> Path:
    """Create and verify the single durable storage root before any live work."""
    root = Path(path).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".write-check-{os.getpid()}"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(
            f"RL collection storage is not writable: {root}. "
            "Ask the directory owner/admin to grant write access; no fallback to home/SSD is used."
        ) from exc
    return root


def validate_collection_config(config: dict[str, Any]) -> dict[str, Any]:
    if not config.get("experiment_id"):
        raise ValueError("config.experiment_id is required")
    datasets = config.get("datasets") or {}
    if not datasets:
        raise ValueError("config.datasets must map dataset IDs to train or heldout")
    normalized_splits: dict[str, str] = {}
    for dataset_id, split in datasets.items():
        if split not in {"train", "heldout"}:
            raise ValueError(f"unsupported split for {dataset_id}: {split}")
        normalized_id = normalized_dataset_id(dataset_id)
        if normalized_id in normalized_splits and normalized_splits[normalized_id] != split:
            raise ValueError(f"parent dataset and branch {dataset_id} cannot cross train/heldout splits")
        normalized_splits[normalized_id] = split
    space = config.get("parameter_space") or {}
    missing = set(STAGES) - set(space)
    if missing:
        raise ValueError(f"parameter_space missing stages: {sorted(missing)}")
    box_sizes = space["extract_box_size"].get("box_size_pix")
    if not isinstance(box_sizes, list) or not box_sizes:
        raise ValueError("extract_box_size.box_size_pix must be a non-empty list")
    if any(not isinstance(size, int) or size < 32 or size > 720 for size in box_sizes):
        raise ValueError("box-size choices must be integer pixels in [32, 720]")
    normalized_datasets = {normalized_dataset_id(key) for key in datasets}
    overrides = config.get("dataset_parameter_space") or {}
    for dataset_id, dataset_space in overrides.items():
        key = normalized_dataset_id(dataset_id)
        if key not in normalized_datasets:
            raise ValueError(f"dataset_parameter_space references undeclared dataset: {dataset_id}")
        if not isinstance(dataset_space, dict):
            raise ValueError(f"dataset_parameter_space.{dataset_id} must be an object")
        unknown_stages = set(dataset_space) - set(STAGES)
        if unknown_stages:
            raise ValueError(f"dataset_parameter_space.{dataset_id} has unknown stages: {sorted(unknown_stages)}")
    for dataset_id in datasets:
        effective_space = parameter_space_for_dataset(config, dataset_id)
        missing = set(STAGES) - set(effective_space)
        if missing:
            raise ValueError(f"parameter_space missing stages for {dataset_id}: {sorted(missing)}")
        box_sizes = effective_space["extract_box_size"].get("box_size_pix")
        if not isinstance(box_sizes, list) or not box_sizes:
            raise ValueError(f"extract_box_size.box_size_pix must be a non-empty list for {dataset_id}")
        if any(not isinstance(size, int) or size < 32 or size > 720 for size in box_sizes):
            raise ValueError(f"box-size choices for {dataset_id} must be integer pixels in [32, 720]")
        for stage in ("blob_picker_diameter", "inspect_picks_threshold"):
            fields = STAGE_PARAMETER_FIELDS[stage]
            for field in fields:
                if field not in effective_space[stage]:
                    raise ValueError(f"parameter_space.{stage}.{field} is required for {dataset_id}")
                bounds = effective_space[stage][field]
                if (
                    not isinstance(bounds, list) or len(bounds) != 2
                    or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bounds)
                    or float(bounds[0]) > float(bounds[1])
                ):
                    raise ValueError(f"{stage}.{field} must be [minimum, maximum] for {dataset_id}")
    limits = config.setdefault("limits", {})
    limits.setdefault("max_trials_per_dataset", 12)
    limits.setdefault("max_trials_total", 100)
    limits.setdefault("max_jobs_per_trial", 4)
    limits.setdefault("max_decisions_per_stage", 2)
    # Older pilot configs carried a particle-count cap. Drop it: the sampling
    # unit is a fixed set of micrographs, not an arbitrary particle ceiling.
    limits.pop("max_input_particles", None)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in limits.values()):
        raise ValueError("all collection limits must be positive integers")
    hard_caps = {
        "max_trials_per_dataset": 12,
        "max_trials_total": 400,
        "max_jobs_per_trial": 4,
        "max_decisions_per_stage": 2,
    }
    if any(limits[key] > ceiling for key, ceiling in hard_caps.items()):
        raise ValueError(f"collection limits may only be lowered from hard caps: {hard_caps}")
    sampling = config.setdefault("sampling", {})
    sampling.setdefault("method", "cryosparc_exposure_sets_tool")
    sampling.setdefault("micrograph_count", FIXED_MICROGRAPH_COUNT)
    sampling.setdefault("randomize", True)
    sampling.setdefault("reuse_subset_across_trials", True)
    if sampling["method"] != "cryosparc_exposure_sets_tool":
        raise ValueError("sampling.method must be cryosparc_exposure_sets_tool")
    if sampling["micrograph_count"] != FIXED_MICROGRAPH_COUNT:
        raise ValueError(f"sampling.micrograph_count is fixed at {FIXED_MICROGRAPH_COUNT}")
    if sampling["randomize"] is not True or sampling["reuse_subset_across_trials"] is not True:
        raise ValueError("sampling must randomize once and reuse the same subset across trials")
    return config


def parameter_space_for_dataset(config: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    """Merge dataset-specific ranges over the experiment-wide parameter defaults."""
    base = config.get("parameter_space") or {}
    key = normalized_dataset_id(dataset_id)
    overrides = config.get("dataset_parameter_space") or {}
    override = next(
        (value for name, value in overrides.items() if normalized_dataset_id(name) == key),
        {},
    )
    merged = {stage: dict(values) for stage, values in base.items()}
    for stage, values in override.items():
        merged.setdefault(stage, {}).update(values)
    return merged


def validate_micrograph_subset(
    subset: dict[str, Any], expected_count: int = FIXED_MICROGRAPH_COUNT,
) -> dict[str, Any]:
    """Validate a stable UID bundle from one Exposure Sets Tool split output."""
    job_uid = subset.get("source_job_uid")
    output_name = subset.get("output_name")
    uids = subset.get("micrograph_uids")
    if not job_uid or not output_name:
        raise ValueError("subset requires source_job_uid and output_name, e.g. split_0")
    if output_name != "split_0":
        raise ValueError("the RL sample must bind the Exposure Sets Tool split_0 output")
    if not isinstance(uids, list) or len(uids) != expected_count:
        raise ValueError(f"subset must contain exactly {expected_count} micrograph UIDs")
    normalized_uids = [str(uid) for uid in uids]
    if any(not uid for uid in normalized_uids) or len(set(normalized_uids)) != expected_count:
        raise ValueError("micrograph_uids must be non-empty and unique")
    fingerprint = hashlib.sha256("\n".join(sorted(normalized_uids)).encode("utf-8")).hexdigest()
    if subset.get("fingerprint") not in (None, fingerprint):
        raise ValueError("subset fingerprint does not match micrograph_uids")
    return {
        "source_job_uid": str(job_uid),
        "output_name": str(output_name),
        "micrograph_uids": normalized_uids,
        "micrograph_count": expected_count,
        "fingerprint": fingerprint,
        "sampling_method": "cryosparc_exposure_sets_tool",
        "randomized_once_and_reused": True,
        **(
            {
                "ctf_job_uid": str(subset["ctf_job_uid"]),
                "ctf_output_name": str(subset.get("ctf_output_name") or "exposures"),
                "ctf_output_count": int(subset["ctf_output_count"])
                if subset.get("ctf_output_count") is not None else None,
                "ctf_retained_micrograph_uids": [
                    str(uid) for uid in subset.get("ctf_retained_micrograph_uids", [])
                ],
                "ctf_dropped_micrograph_uids": [
                    str(uid) for uid in subset.get("ctf_dropped_micrograph_uids", [])
                ],
            }
            if subset.get("ctf_job_uid") else {}
        ),
    }


class CollectionStore:
    """SQLite-backed, resumable ledger for model decisions and measured outcomes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                    dataset_id TEXT NOT NULL,
                    split TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id TEXT NOT NULL REFERENCES trials(trial_id),
                    stage TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    job_uid TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    trial_id TEXT PRIMARY KEY REFERENCES trials(trial_id),
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

    def initialize_experiment(self, config: dict[str, Any]) -> None:
        config = validate_collection_config(config)
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO experiments VALUES (?, ?, ?)",
                (config["experiment_id"], json.dumps(config), utc_now()),
            )
            stored = db.execute(
                "SELECT config_json FROM experiments WHERE experiment_id = ?",
                (config["experiment_id"],),
            ).fetchone()
            if json.loads(stored["config_json"]) != config:
                raise ValueError("experiment_id already exists with a different config")

    def config(self, experiment_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT config_json FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown experiment: {experiment_id}")
        return json.loads(row["config_json"])

    def create_trial(
        self,
        experiment_id: str,
        dataset_id: str,
        metadata: dict[str, Any] | None = None,
        trial_id: str | None = None,
    ) -> str:
        config = self.config(experiment_id)
        key = normalized_dataset_id(dataset_id)
        dataset_map = {normalized_dataset_id(item): split for item, split in config["datasets"].items()}
        if key not in dataset_map:
            raise ValueError(f"dataset {dataset_id} is not declared in the experiment config")
        if trial_id:
            try:
                existing = self.trial(trial_id)
            except ValueError:
                existing = None
            if existing is not None:
                if (existing["experiment_id"] != experiment_id
                        or normalized_dataset_id(existing["dataset_id"]) != key):
                    raise ValueError(f"trial_id {trial_id} is already bound to another experiment/dataset")
                return trial_id
        limits = config["limits"]
        with self._connect() as db:
            total = db.execute(
                "SELECT COUNT(*) AS n FROM trials WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()["n"]
            existing_dataset_ids = db.execute(
                "SELECT dataset_id FROM trials WHERE experiment_id = ?", (experiment_id,)
            ).fetchall()
            per_dataset = sum(
                normalized_dataset_id(row["dataset_id"]) == key for row in existing_dataset_ids
            )
        if total >= limits["max_trials_total"]:
            raise ValueError("experiment trial budget exhausted")
        if per_dataset >= limits["max_trials_per_dataset"]:
            raise ValueError(f"trial budget exhausted for dataset {dataset_id}")
        trial_id = trial_id or str(uuid.uuid4())
        now = utc_now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (trial_id, experiment_id, dataset_id, dataset_map[key], "active",
                 json.dumps(metadata or {}, ensure_ascii=False), now, now),
            )
        return trial_id

    def bind_micrograph_subset(self, trial_id: str, subset: dict[str, Any]) -> dict[str, Any]:
        """Bind a verified Exposure Sets Tool output to a trial before submission."""
        trial = self.trial(trial_id)
        config = self.config(trial["experiment_id"])
        checked = validate_micrograph_subset(subset, config["sampling"]["micrograph_count"])
        with self._connect() as db:
            row = db.execute(
                "SELECT metadata_json FROM trials WHERE trial_id = ?", (trial_id,)
            ).fetchone()
            metadata = json.loads(row["metadata_json"])
            prior = metadata.get("fixed_micrograph_subset")
            if prior and prior["fingerprint"] != checked["fingerprint"]:
                raise ValueError("trial already has a different fixed micrograph subset")
            metadata["fixed_micrograph_subset"] = checked
            db.execute(
                "UPDATE trials SET metadata_json = ?, updated_at = ? WHERE trial_id = ?",
                (json.dumps(metadata, ensure_ascii=False), utc_now(), trial_id),
            )
        return checked

    def record_event(
        self, trial_id: str, stage: str, event_type: str, payload: dict[str, Any], job_uid: str | None = None,
    ) -> None:
        if stage not in EVENT_STAGES:
            raise ValueError(f"unsupported event stage: {stage}")
        with self._connect() as db:
            trial = db.execute(
                "SELECT t.experiment_id FROM trials t WHERE t.trial_id = ?", (trial_id,)
            ).fetchone()
            if trial is None:
                raise ValueError(f"unknown trial: {trial_id}")
            limits = self.config(trial["experiment_id"])["limits"]
            if event_type in {"model_decision", "model_decision_invalid"}:
                count = db.execute(
                    "SELECT COUNT(*) AS n FROM events WHERE trial_id = ? AND stage = ? "
                    "AND event_type IN ('model_decision', 'model_decision_invalid') "
                    "AND event_id > COALESCE((SELECT MAX(event_id) FROM events "
                    "WHERE trial_id = ? AND event_type = 'retry_started'), 0)",
                    (trial_id, stage, trial_id),
                ).fetchone()["n"]
                if count >= limits["max_decisions_per_stage"]:
                    raise ValueError(f"decision budget exhausted for {stage}")
            if event_type == "job_status" and job_uid:
                jobs = db.execute(
                    "SELECT COUNT(DISTINCT job_uid) AS n FROM events WHERE trial_id = ? AND event_type = 'job_status'",
                    (trial_id,),
                ).fetchone()["n"]
                known = db.execute(
                    "SELECT 1 FROM events WHERE trial_id = ? AND event_type = 'job_status' AND job_uid = ?",
                    (trial_id, job_uid),
                ).fetchone()
                if not known and jobs >= limits["max_jobs_per_trial"]:
                    raise ValueError("per-trial CryoSPARC job budget exhausted")
            db.execute(
                "INSERT INTO events (trial_id, stage, event_type, job_uid, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (trial_id, stage, event_type, job_uid, json.dumps(payload, ensure_ascii=False), utc_now()),
            )
            db.execute("UPDATE trials SET updated_at = ? WHERE trial_id = ?", (utc_now(), trial_id))

    def trial(self, trial_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
            if row is None:
                raise ValueError(f"unknown trial: {trial_id}")
            events = db.execute(
                "SELECT stage, event_type, job_uid, payload_json, created_at FROM events "
                "WHERE trial_id = ? ORDER BY event_id", (trial_id,),
            ).fetchall()
            outcome = db.execute("SELECT payload_json FROM outcomes WHERE trial_id = ?", (trial_id,)).fetchone()
        return {
            "trial_id": row["trial_id"], "experiment_id": row["experiment_id"],
            "dataset_id": row["dataset_id"], "split": row["split"], "status": row["status"],
            "metadata": json.loads(row["metadata_json"]),
            "events": [{**dict(event), "payload": json.loads(event["payload_json"])} for event in events],
            "outcome": json.loads(outcome["payload_json"]) if outcome else None,
        }

    def completed_feedback(self, experiment_id: str, split: str = "train") -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT t.trial_id, t.dataset_id, t.split, o.payload_json FROM trials t "
                "JOIN outcomes o ON o.trial_id = t.trial_id "
                "WHERE t.experiment_id = ? AND t.split = ? ORDER BY t.created_at",
                (experiment_id, split),
            ).fetchall()
        result = []
        for row in rows:
            outcome = json.loads(row["payload_json"])
            if outcome.get("episode_status") == "failed":
                continue
            episode = self.trial(row["trial_id"])
            result.append({
                "trial_id": row["trial_id"],
                "dataset_id": row["dataset_id"],
                "decisions": [
                    {"stage": event["stage"], "decision": event["payload"].get("decision")}
                    for event in episode["events"] if event["event_type"] == "model_decision"
                ],
                "outcome": outcome,
            })
        return result

    def record_outcome(self, trial_id: str, outcome: dict[str, Any]) -> None:
        with self._connect() as db:
            row = db.execute("SELECT status FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
            if row is None:
                raise ValueError(f"unknown trial: {trial_id}")
            db.execute(
                "INSERT OR REPLACE INTO outcomes VALUES (?, ?, ?)",
                (trial_id, json.dumps(outcome, ensure_ascii=False), utc_now()),
            )
            db.execute(
                "UPDATE trials SET status = ?, updated_at = ? WHERE trial_id = ?",
                ("failed" if outcome.get("episode_status") == "failed" else "completed",
                 utc_now(), trial_id),
            )

    def export_episodes(self, experiment_id: str, output_path: str | Path, split: str = "train") -> int:
        with self._connect() as db:
            rows = db.execute(
                "SELECT trial_id FROM trials WHERE experiment_id = ? AND split = ? ORDER BY created_at",
                (experiment_id, split),
            ).fetchall()
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with target.open("w", encoding="utf-8") as stream:
            for row in rows:
                episode = self.trial(row["trial_id"])
                if episode["outcome"] is None:
                    continue
                stream.write(json.dumps(episode, ensure_ascii=False) + "\n")
                count += 1
        return count


def validate_stage_decision(stage: str, decision: dict[str, Any], search_space: dict[str, Any]) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unsupported stage: {stage}")
    if decision.get("decision") != STAGE_ACTIONS[stage]:
        raise ValueError(f"decision must be {STAGE_ACTIONS[stage]} for stage {stage}")
    parameters = decision.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("decision.parameters must be an object")
    required = set(STAGE_PARAMETER_FIELDS[stage])
    if stage == "blob_picker_diameter":
        required = {"diameter"}
    if stage == "inspect_picks_threshold":
        required = {"ncc_score_min", "local_power_min"}
    missing = required - parameters.keys()
    allowed_fields = set(STAGE_PARAMETER_FIELDS[stage])
    if missing:
        raise ValueError(f"decision is missing parameters: {sorted(missing)}")
    if parameters.keys() - allowed_fields:
        raise ValueError(f"unsupported decision parameters: {sorted(parameters.keys() - allowed_fields)}")
    for field, value in parameters.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field} must be numeric")
        if stage == "extract_box_size":
            if int(value) != value or int(value) not in search_space[stage][field]:
                raise ValueError(f"{field} must be one of {search_space[stage][field]}")
        elif field in search_space[stage]:
            low, high = map(float, search_space[stage][field])
            if not low <= float(value) <= high:
                raise ValueError(f"{field}={value} is outside configured bounds [{low}, {high}]")
    if stage == "blob_picker_diameter" and parameters.get("diameter_max") is not None:
        if float(parameters["diameter_max"]) < float(parameters["diameter"]):
            raise ValueError("diameter_max must be greater than or equal to diameter")
    if stage == "inspect_picks_threshold" and parameters.get("local_power_max") is not None:
        if float(parameters["local_power_max"]) < float(parameters["local_power_min"]):
            raise ValueError("local_power_max must be greater than or equal to local_power_min")
    return decision


def to_v2_execution_decision(stage: str, decision: dict[str, Any]) -> dict[str, Any]:
    """Translate the compact parameter proposal to the MCP V2 executor schema."""
    params = dict(decision["parameters"])
    aliases = {
        "ncc_score_min": "ncc_score_thresh",
        "local_power_min": "lpower_thresh_min",
        "local_power_max": "lpower_thresh_max",
    }
    for source, target in aliases.items():
        if source in params:
            params[target] = params.pop(source)
    action = STAGE_ACTIONS[stage]
    return {
        "schema_version": "2.0",
        "decision_type": "forward",
        "action": action,
        "job_type": action,
        "parameters": params,
        "reason": decision.get("reason", "Bounded visual self-play parameter proposal."),
        "confidence": decision.get("confidence", 0.5),
        "risk_flags": decision.get("risk_flags", []),
        "evidence": decision.get("evidence", []),
    }


def compute_class2d_feedback(
    class_statistics: list[dict[str, Any]],
    input_particle_count: int,
    *,
    min_usable_particles: int = 1000,
    min_particle_coverage: float = 0.5,
    baseline_resolution_A: float | None = None,
    reference_projection_metrics: dict[str, Any] | None = None,
    cryosift_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return transparent, vector-valued feedback; do not hide tradeoffs in weights."""
    valid = []
    for item in class_statistics:
        count = item.get("particle_count", item.get("num_particles"))
        resolution = item.get("resolution_A", item.get("res_A"))
        if (
            isinstance(count, (int, float)) and not isinstance(count, bool) and count > 0
            and isinstance(resolution, (int, float)) and not isinstance(resolution, bool)
            and 0 < resolution < float("inf")
        ):
            valid.append((float(count), float(resolution)))
    particle_count = sum(count for count, _ in valid)
    weighted_resolution = (
        sum(count * resolution for count, resolution in valid) / particle_count
        if particle_count else None
    )
    coverage = particle_count / input_particle_count if input_particle_count > 0 else None
    eligible_by_class2d = bool(
        weighted_resolution is not None
        and particle_count >= min_usable_particles
        and coverage is not None
        and coverage >= min_particle_coverage
    )
    if baseline_resolution_A is not None and (
        not math.isfinite(float(baseline_resolution_A)) or float(baseline_resolution_A) <= 0
    ):
        raise ValueError("baseline_resolution_A must be a positive finite value")
    cryosift_metrics: dict[str, Any] = {"status": "not_run"}
    if cryosift_result is not None:
        if cryosift_result.get("success"):
            selected_ids = {
                int(value) for value in cryosift_result.get("selected_class_ids", [])
                if isinstance(value, (int, float, str)) and str(value).lstrip("-+").isdigit()
            }
            class_by_id = {
                int(item.get("class_id", index)): item
                for index, item in enumerate(class_statistics)
                if isinstance(item.get("class_id", index), (int, float))
            }
            selected_particles = sum(
                max(0, int(class_by_id[class_id].get("particle_count", class_by_id[class_id].get("num_particles", 0)) or 0))
                for class_id in selected_ids if class_id in class_by_id
            )
            cryosift_metrics = {
                "status": "completed",
                "threshold": cryosift_result.get("threshold"),
                "score_direction": cryosift_result.get("score_direction", "lower_is_better"),
                "selected_class_ids": sorted(selected_ids),
                "selected_class_count": len(selected_ids),
                "selected_particle_count": selected_particles,
                "input_particle_count": int(input_particle_count),
                "selected_particle_fraction": (
                    selected_particles / input_particle_count if input_particle_count > 0 else None
                ),
                "class_scores": cryosift_result.get("class_scores", []),
            }
        else:
            cryosift_metrics = {
                "status": cryosift_result.get("status", "failed"),
                "error": cryosift_result.get("message") or cryosift_result.get("stderr_tail"),
            }
    projection_metrics = reference_projection_metrics or {"status": "not_run"}
    cryosift_guard_required = cryosift_result is not None
    cryosift_guard_passed = (
        cryosift_metrics.get("status") == "completed"
        and cryosift_metrics.get("selected_particle_count", 0) >= min_usable_particles
    ) if cryosift_guard_required else None
    eligible = eligible_by_class2d and (
        not cryosift_guard_required or bool(cryosift_guard_passed)
    )
    reasons = []
    if not eligible_by_class2d:
        reasons.append("insufficient scored-particle count, coverage, or valid class resolution")
    if cryosift_guard_required and not cryosift_guard_passed:
        reasons.append("CryoSift-recommended classes contain fewer than min_usable_particles")
    reward = None
    if eligible:
        reward = (
            float(baseline_resolution_A) - weighted_resolution
            if baseline_resolution_A is not None else -weighted_resolution
        )
    return {
        "metric_version": "class2d-feedback-0.2",
        "valid_class_count": len(valid),
        "input_particle_count": int(input_particle_count),
        "particles_in_scored_classes": int(round(particle_count)),
        "particle_coverage": coverage,
        "particle_weighted_resolution_A": weighted_resolution,
        "reference_projection": projection_metrics,
        "view_diversity": projection_metrics.get("view_diversity"),
        "cryosift_retention": cryosift_metrics,
        "cryosift_particle_count_guard": {
            "required": cryosift_guard_required,
            "minimum_selected_particles": min_usable_particles if cryosift_guard_required else None,
            "passed": cryosift_guard_passed,
        },
        "baseline_resolution_A": baseline_resolution_A,
        "resolution_reward": reward,
        "reward_mode": "paired_improvement_A" if baseline_resolution_A is not None else "negative_absolute_A",
        "reward_direction": "higher_is_better; paired reward is baseline minus current weighted resolution",
        "cross_dataset_comparable": baseline_resolution_A is not None,
        "eligible_for_policy_update": eligible,
        "eligibility_reason": "; ".join(reasons) if reasons else None,
        "reward_components": {
            "resolution": "particle-count-weighted Class 2D resolution; lower Angstrom value is better",
            "coverage": "reported separately and used only as an eligibility guard, preventing reward by retaining very few particles",
            "reference_projection": "particle-weighted morphology consistency against offline reference-map projections; not a direct calibrated truncation probability",
            "view_diversity": "particle-weighted entropy of best-matching reference projection directions",
            "cryosift_retention": "CryoSift-recommended class particle count divided by the input particle count; lower CryoSift scores indicate better classes",
        },
        "multi_objective_note": "Metrics are reported separately; no uncalibrated scalar combination is used.",
    }


def collect_images(visual_context: dict[str, Any] | None) -> list[str]:
    def data_url(item: dict[str, Any]) -> str | None:
        if item.get("data_url"):
            return item["data_url"]
        local_path = item.get("local_path")
        if local_path:
            path = Path(local_path)
            if path.is_file():
                mime_type = item.get("mime_type") or "image/png"
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                return f"data:{mime_type};base64,{encoded}"
        return None

    images = []
    for key in ("contact_sheet", "score_scatter"):
        item = (visual_context or {}).get(key) or {}
        value = data_url(item)
        if value:
            images.append(value)
    for item in (visual_context or {}).get("evidence_images") or []:
        value = data_url(item)
        if value:
            images.append(value)
    return images


def _model_worker_project_dir() -> Path:
    """Directory that owns scripts/model_generate_vision_once.py."""
    return Path(__file__).resolve().parent


def compact_visual_context(visual_context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in visual_context.items()
        if key not in {"contact_sheet", "score_scatter", "evidence_images"}
    } | {
        "evidence_image_descriptions": visual_context.get("evidence_image_descriptions", []),
        "score_scatter": {
            key: value for key, value in (visual_context.get("score_scatter") or {}).items()
            if key != "data_url"
        },
    }


def visual_evidence_record(visual_context: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint the actual attachments without persisting inline image data."""
    images = []
    for key in ("contact_sheet", "score_scatter"):
        item = visual_context.get(key) or {}
        if item.get("data_url") or item.get("local_path"):
            images.append(item)
    images.extend(item for item in visual_context.get("evidence_images") or []
                  if item.get("data_url") or item.get("local_path"))
    attachments = []
    for index, item in enumerate(images):
        path = Path(item["local_path"]) if item.get("local_path") else None
        local_bytes = path.read_bytes() if path and path.is_file() else None
        if item.get("data_url"):
            header, value = item["data_url"].split(",", 1)
            if ";base64" not in header:
                raise ValueError("RL evidence fingerprints require base64 image data URLs")
            data = base64.b64decode(value)
        else:
            data = local_bytes
        attachments.append({"index": index, "role": item.get("role"),
                            "local_path": str(path) if path else None,
                            "sha256": hashlib.sha256(data).hexdigest() if data is not None else None,
                            "local_file_matches_attachment": local_bytes == data if local_bytes is not None else False})
    return {
        "kind": visual_context.get("kind"), "source": visual_context.get("source"),
        "visual_evidence_version": visual_context.get("visual_evidence_version", "legacy_unversioned"),
        "image_count": len(attachments), "attachments": attachments,
        "image_roles": [item.get("role") for item in visual_context.get("evidence_images") or []],
    }


def durable_messages(messages: list[dict[str, Any]], visual_context: dict[str, Any]) -> list[dict[str, Any]]:
    """Persist local image paths instead of duplicating base64 data in SQLite."""
    paths = []
    for key in ("contact_sheet", "score_scatter"):
        image = visual_context.get(key) or {}
        if image.get("data_url") or image.get("local_path"):
            paths.append(image.get("local_path"))
    for image in visual_context.get("evidence_images") or []:
        if image.get("data_url") or image.get("local_path"):
            paths.append(image.get("local_path"))
    path_iter = iter(paths)
    stored = json.loads(json.dumps(messages))
    for message in stored:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if part.get("type") == "image_url":
                message["content"][index] = {"type": "image", "path": next(path_iter, None)}
    return stored


def _model_decision(messages: list[dict[str, Any]], args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.backend == "api":
        result = run_openai_compatible_model(
            messages,
            api_base=args.api_base,
            api_key=resolve_api_key(api_key_env=args.api_key_env),
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
        return result["raw_text"], result
    persistent_worker = getattr(args, "model_worker", None)
    if persistent_worker is not None:
        try:
            result = persistent_worker.generate(
                str(uuid.uuid4()), messages, args.max_new_tokens, args.temperature,
            )
        except (RuntimeError, BrokenPipeError):
            process = getattr(persistent_worker, "process", None)
            if process is None or process.poll() is None:
                raise
            persistent_worker.close()
            restarts = getattr(persistent_worker, "collection_restarts", 0) + 1
            persistent_worker.collection_restarts = restarts
            persistent_worker.stderr_path = persistent_worker.stderr_path.with_name(
                f"model_worker.restart-{restarts}.stderr.log"
            )
            persistent_worker.start()
            result = persistent_worker.generate(
                str(uuid.uuid4()), messages, args.max_new_tokens, args.temperature,
            )
            result["worker_restarted"] = True
        return result.get("raw_text", ""), result
    from autonomous_mcp_closed_loop import ModelWorker

    worker_args = SimpleNamespace(
        vision=True,
        model_python=args.model_python,
        project_dir=str(_model_worker_project_dir()),
        base_model=args.base_model,
        adapter=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        model_srun_prefix=args.model_srun_prefix,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    worker = ModelWorker(worker_args, Path(args.output_dir) / "model_worker.stderr.log")
    try:
        worker.start()
        result = worker.generate(str(uuid.uuid4()), messages, args.max_new_tokens, args.temperature)
        return result.get("raw_text", ""), result
    finally:
        worker.close()


def exploration_context(store, config, trial, stage):
    """Keep same-dataset comparisons visible even after many other trials."""
    exploration = config.get("exploration") or {}
    if not exploration or trial["split"] != "train":
        return {}
    sources = list(dict.fromkeys([
        config["experiment_id"], *exploration.get("feedback_experiment_ids", []),
    ]))
    episodes = [
        item for source in sources for item in store.completed_feedback(source, split="train")
        if normalized_dataset_id(item["dataset_id"]) == normalized_dataset_id(trial["dataset_id"])
        and item["trial_id"] != trial["trial_id"]
    ]
    return {
        "same_dataset_prior_feedback": episodes,
        "exploration_guidance": exploration.get("guidance"),
        "previous_stage_parameters": [
            decision["decision"].get("parameters", {})
            for item in episodes for decision in item["decisions"]
            if decision["stage"] == stage and isinstance(decision.get("decision"), dict)
        ],
    }


def decide_stage(args: argparse.Namespace) -> dict[str, Any]:
    store = CollectionStore(args.db)
    config = store.config(args.experiment_id)
    trial_id = args.trial_id or store.create_trial(args.experiment_id, args.dataset_id)
    trial = store.trial(trial_id)
    if trial["experiment_id"] != args.experiment_id or normalized_dataset_id(trial["dataset_id"]) != normalized_dataset_id(args.dataset_id):
        raise ValueError("trial does not belong to the selected experiment/dataset")
    split = trial["split"]
    prior_stage_decisions = sum(
        event["event_type"] in {"model_decision", "model_decision_invalid"}
        and event["stage"] == args.stage
        for event in current_attempt_events(trial)
    )
    if prior_stage_decisions >= config["limits"]["max_decisions_per_stage"]:
        raise ValueError(f"decision budget exhausted for {args.stage}")
    history = [
        item for item in store.completed_feedback(args.experiment_id, split="train")
        if split == "train" or normalized_dataset_id(item["dataset_id"]) != normalized_dataset_id(args.dataset_id)
    ]
    parameter_space = parameter_space_for_dataset(config, args.dataset_id)
    visual_context = read_json(args.visual_context_file) if args.visual_context_file else {}
    context = read_json(args.context_file)
    context.update(exploration_context(store, config, trial, args.stage))
    context.update({
        "experiment_id": args.experiment_id,
        "dataset_id": args.dataset_id,
        "dataset_split": split,
        "sampling": config["sampling"],
        "fixed_micrograph_subset": trial["metadata"].get("fixed_micrograph_subset"),
        "search_space": parameter_space[args.stage],
        "prior_train_feedback": history[-20:],
        "previous_invalid_decisions": [
            {
                "stage": event["stage"],
                "raw_text": event["payload"].get("raw_text"),
                "error": event["payload"].get("error"),
            }
            for event in trial["events"]
            if event["event_type"] == "model_decision_invalid"
            and event["stage"] == args.stage
        ],
        "visual_evidence_metadata": compact_visual_context(visual_context),
        "trial_history": [{
            "stage": event["stage"],
            "event_type": event["event_type"],
            "job_uid": event.get("job_uid"),
            "decision": event["payload"].get("decision"),
            "status": event["payload"].get("status"),
        } for event in trial["events"]],
    })
    messages = build_stage_decision_messages(
        args.stage, context, image_data_urls=collect_images(visual_context),
    )
    raw_text, generation = _model_decision(messages, args)
    try:
        decision = parse_model_decision_text(raw_text)
        validate_stage_decision(args.stage, decision, parameter_space)
    except Exception as exc:
        response = generation.get("raw_response") or {}
        choices = response.get("choices") or []
        first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        store.record_event(trial_id, args.stage, "model_decision_invalid", {
            "raw_text": raw_text,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_generation": {
                "backend": args.backend,
                "model": args.model if args.backend == "api" else args.base_model,
                "endpoint": generation.get("endpoint"),
                "attempts": generation.get("attempts"),
                "finish_reason": first_choice.get("finish_reason"),
                "usage": response.get("usage"),
                "assistant_message": generation.get("assistant_message"),
            },
            "messages": durable_messages(messages, visual_context),
            "visual_evidence": visual_evidence_record(visual_context),
        })
        raise
    payload = {
        "decision": decision,
        "raw_text": raw_text,
        "messages": durable_messages(messages, visual_context),
        "model_generation": {
            "backend": args.backend,
            "model": args.model if args.backend == "api" else args.base_model,
            "attempts": generation.get("attempts"),
            "worker_request_number": generation.get("worker_request_number"),
        },
        "visual_evidence": visual_evidence_record(visual_context),
    }
    store.record_event(trial_id, args.stage, "model_decision", payload)
    return {"trial_id": trial_id, "stage": args.stage, "split": split, **payload}


def require_expected_stage_source(trial: dict[str, Any], stage: str, current_node: str) -> None:
    """Reject live submissions that drift away from the fixed subset lineage."""
    subset = trial["metadata"].get("fixed_micrograph_subset") or {}
    if stage == "blob_picker_diameter":
        expected = subset.get("ctf_job_uid") or subset.get("source_job_uid")
        if not expected or current_node != expected:
            raise ValueError(f"Blob Picker must start from the bound CTF/exposure Job {expected!r}")
        return
    previous_stage = {
        "inspect_picks_threshold": "blob_picker_diameter",
        "extract_box_size": "inspect_picks_threshold",
    }[stage]
    matching = [
        event for event in trial["events"]
        if event["stage"] == previous_stage and event["event_type"] == "job_status"
        and event.get("job_uid")
    ]
    if not matching or matching[-1]["payload"].get("status") != "completed":
        raise ValueError(f"{stage} requires a completed {previous_stage} Job in this trial")
    expected = matching[-1]["job_uid"]
    if current_node != expected:
        raise ValueError(f"{stage} must continue from this trial's completed Job {expected}")


def validate_ctf_source(
    project: Any,
    job_uid: str,
    workspace_uid: str,
    expected_uids: list[str],
    source_job_uid: str | None = None,
    expected_output_count: int | None = None,
) -> Any:
    """Require a completed Patch CTF result for this trial's fixed subset."""
    job = project.find_job(job_uid)
    if str(job.status).lower() != "completed":
        raise ValueError(f"Patch CTF Job {job_uid} is not completed (status={job.status})")
    model = job.model
    workspace_uids = set(getattr(model, "workspace_uids", []) or [])
    if workspace_uid not in workspace_uids:
        raise ValueError(f"Patch CTF Job {job_uid} is not in workspace {workspace_uid}")
    if str(getattr(job, "type", "")) != "patch_ctf_estimation_multi":
        raise ValueError(f"Job {job_uid} is not a Patch CTF Job")
    if source_job_uid:
        spec = model.spec.model_dump(mode="json")
        connections = ((spec.get("inputs") or {}).get("exposures") or {}).get("connections") or []
        source_uids = {
            str(connection.get("job_uid"))
            for connection in (connections or [])
            if isinstance(connection, dict) and connection.get("job_uid")
        }
        if source_job_uid not in source_uids:
            raise ValueError(f"Patch CTF Job {job_uid} is not connected to Exposure Sets Job {source_job_uid}")
    exposures = job.load_output("exposures")
    output_uids = [str(row["uid"]) for row in exposures.rows()]
    source_uids = {str(uid) for uid in expected_uids}
    output_set = set(output_uids)
    if not output_set or len(output_set) != len(output_uids) or output_set - source_uids:
        raise ValueError(
            f"Patch CTF Job {job_uid} output is empty, duplicated, or outside the fixed subset"
        )
    if expected_output_count is not None and len(output_set) != expected_output_count:
        raise ValueError(
            f"Patch CTF Job {job_uid} output has {len(output_set)} exposures; "
            f"manifest records {expected_output_count}"
        )
    return job


async def execute_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Run the stored decision through MCP validation; submission requires --submit."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from autonomous_mcp_closed_loop import call_tool_json

    store = CollectionStore(args.db)
    trial = store.trial(args.trial_id)
    config = store.config(trial["experiment_id"])
    subset = trial["metadata"].get("fixed_micrograph_subset")
    if args.submit:
        if not subset:
            raise ValueError(
                "Live submission requires a verified 20-micrograph Exposure Sets Tool output. "
                "Create one randomized split per dataset, then bind its split_0 UID list with bind-subset."
            )
        validate_micrograph_subset(subset, config["sampling"]["micrograph_count"])
        require_expected_stage_source(trial, args.stage, args.current_node)
    decisions = [
        event["payload"]["decision"] for event in trial["events"]
        if event["stage"] == args.stage and event["event_type"] == "model_decision"
    ]
    if not decisions:
        raise ValueError(f"trial has no valid model decision for {args.stage}")
    execution_decision = to_v2_execution_decision(args.stage, decisions[-1])
    project_dir = Path(__file__).resolve().parents[1]
    server_env = os.environ.copy()
    server_env["CRYOAGENT_VISION_CACHE_DIR"] = str(Path(args.output_dir) / "vision_inputs")
    server = StdioServerParameters(
        command=args.server_python,
        args=[str(project_dir / "cryosparc_agent" / "cryosparc_mcp_server.py")],
        cwd=str(project_dir),
        env=server_env,
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await call_tool_json(session, "execute_v2_model_decision", {
                "decision": execution_decision,
                "project_uid": args.project,
                "workspace_uid": args.workspace,
                "current_node_id": args.current_node,
                "dataset_info": config.get("dataset_info", {}).get(trial["dataset_id"], {}),
                "dry_run": not args.submit,
            })
    store.record_event(trial["trial_id"], args.stage, "execution", {
        "dry_run": not args.submit,
        "decision": execution_decision,
        "fixed_micrograph_subset": subset,
        "result": result,
    })
    if args.submit:
        execution_result = result.get("execution_result") or {}
        jobs = execution_result.get("execution_results") or []
        for job in jobs:
            if not job.get("job_uid"):
                continue
            success = bool(job.get("success"))
            store.record_event(
                trial["trial_id"], args.stage, "job_status",
                {"status": job.get("status", "queued" if success else "failed"),
                 "submitted": True, "success": success,
                 "error": job.get("error")},
                job_uid=job.get("job_uid"),
            )
    return {"trial_id": args.trial_id, "dry_run": not args.submit, "result": result}


def _event_job(trial: dict[str, Any], stage: str) -> dict[str, Any] | None:
    matches = [
        event for event in trial["events"]
        if event["stage"] == stage and event["event_type"] == "job_status" and event.get("job_uid")
    ]
    return matches[-1] if matches else None


def _existing_decision(trial: dict[str, Any], stage: str) -> dict[str, Any] | None:
    matches = [
        event["payload"]["decision"] for event in trial["events"]
        if event["stage"] == stage and event["event_type"] == "model_decision"
    ]
    return matches[-1] if matches else None


def _execution_job_to_recover(trial: dict[str, Any], stage: str) -> dict[str, Any] | None:
    """Close the narrow crash window after enqueue succeeded but before UID journaling."""
    matches = [
        event for event in trial["events"]
        if event["stage"] == stage and event["event_type"] == "execution"
        and not event["payload"].get("dry_run")
    ]
    if not matches:
        return None
    result = matches[-1]["payload"].get("result") or {}
    if not result.get("success"):
        return None
    created = _created_job(result)
    if not created or not created.get("success"):
        return None
    return created


def _durable_visual_context(context: dict[str, Any]) -> dict[str, Any]:
    """Keep local image references and metadata, never inline base64 in run files."""
    durable = json.loads(json.dumps(context))
    for key in ("contact_sheet", "score_scatter"):
        item = durable.get(key)
        if isinstance(item, dict):
            item.pop("data_url", None)
    for item in durable.get("evidence_images") or []:
        if isinstance(item, dict):
            item.pop("data_url", None)
    return durable


def _class2d_decision(config: dict[str, Any]) -> dict[str, Any]:
    parameters = dict((config.get("execution") or {}).get("class_2d_parameters") or {})
    parameters.setdefault("class2D_K", 50)
    parameters.setdefault("compute_num_gpus", 1)
    parameters.setdefault("random_seed", 42)
    return {
        "schema_version": "2.0",
        "decision_type": "forward",
        "action": "class_2D_new",
        "job_type": "class_2D_new",
        "parameters": parameters,
        "reason": "Fixed evaluation stage: quantify the selected 20-micrograph subset with Class 2D.",
        "confidence": 1.0,
        "risk_flags": [],
        "evidence": ["Class 2D parameters are fixed by the collection experiment config."],
    }


def _created_job(result: dict[str, Any]) -> dict[str, Any] | None:
    execution = result.get("execution_result") or {}
    jobs = execution.get("execution_results") or []
    for item in jobs:
        if item.get("job_uid"):
            return item
    return None


async def _submit_decision(
    session: Any,
    store: CollectionStore,
    trial_id: str,
    stage: str,
    decision: dict[str, Any],
    *,
    project: str,
    workspace: str,
    current_node: str,
    dataset_info: dict[str, Any],
    submit: bool,
) -> dict[str, Any]:
    from autonomous_mcp_closed_loop import call_tool_json

    if stage not in (*STAGES, "class_2d"):
        raise ValueError(f"Unapproved collection stage: {stage}")

    result = await call_tool_json(session, "execute_v2_model_decision", {
        "decision": decision,
        "project_uid": project,
        "workspace_uid": workspace,
        "current_node_id": current_node,
        "dataset_info": dataset_info,
        "dry_run": not submit,
        "allow_approval_required_create": bool(submit),
    })
    store.record_event(trial_id, stage, "execution", {
        "dry_run": not submit,
        "decision": decision,
        "current_node_id": current_node,
        "result": result,
    })
    if not submit:
        return {"dry_run": True, "result": result}
    created = _created_job(result)
    if not result.get("success") or not created or not created.get("success"):
        store.record_event(trial_id, stage, "submission_failed", {
            "result": result,
            "reason": "MCP execution did not return a successfully queued Job UID.",
        })
        raise RuntimeError(f"CryoSPARC submission failed for {stage}: {result}")
    status = str(created.get("status") or "queued").lower()
    if status not in {"queued", "running", "completed", "failed", "aborted"}:
        status = "queued"
    store.record_event(trial_id, stage, "job_status", {
        "status": status,
        "submitted": True,
        "success": True,
        "parameters": decision.get("parameters", {}),
        "execution_result": created,
    }, job_uid=str(created["job_uid"]))
    return {"dry_run": False, "job_uid": str(created["job_uid"]), "result": result}


async def _wait_for_trial_job(
    session: Any,
    store: CollectionStore,
    trial_id: str,
    stage: str,
    job_uid: str,
    *,
    project: str,
    workspace: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
    retry_count: int,
) -> dict[str, Any]:
    from autonomous_mcp_closed_loop import wait_for_job_result_with_retries

    result = await wait_for_job_result_with_retries(
        session,
        {
            "project_uid": project,
            "workspace_uid": workspace,
            "job_uid": job_uid,
            "timeout_seconds": timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
            "include_next_candidates": False,
        },
        retries=retry_count,
        retry_delay_seconds=10,
    )
    status = str(result.get("status") or result.get("job_status") or "").lower()
    if result.get("ready_for_model") and status in {"completed", "complete", "success", "succeeded"}:
        normalized = "completed"
    elif status in {"failed", "aborted", "killed", "error"}:
        normalized = "failed" if status != "aborted" else "aborted"
    else:
        normalized = "running"
    store.record_event(trial_id, stage, "job_status", {
        "status": normalized,
        "wait_result": result,
    }, job_uid=job_uid)
    if normalized != "completed":
        failure_outcome = {
            "metric_version": "class2d-feedback-0.1",
            "episode_status": "failed",
            "eligible_for_policy_update": False,
            "resolution_reward": None,
            "failure": {
                "stage": stage, "job_uid": job_uid,
                "status": normalized, "wait_result": result,
            },
            "reward_direction": "failed episode retained without a numeric resolution score",
        }
        store.record_outcome(trial_id, failure_outcome)
        trial = store.trial(trial_id)
        for split in ("train", "heldout"):
            store.export_episodes(
                trial["experiment_id"],
                store.path.parent / "trajectories" / f"{trial['experiment_id']}_{split}.jsonl",
                split=split,
            )
        raise RuntimeError(
            f"Job {job_uid} did not complete successfully (reported status={status!r}, "
            f"ready_for_model={result.get('ready_for_model')}). See trial event log."
        )
    return result


async def run_trial(args: argparse.Namespace) -> dict[str, Any]:
    """Run/resume Blob -> Inspect -> Extract -> Class 2D, keeping all artifacts local to data_root."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from autonomous_mcp_closed_loop import call_tool_json

    root = ensure_writable_data_root(args.data_root)
    db_path = root / "collection.sqlite"
    config = validate_collection_config(read_json(args.config))
    write_json(root / "experiments" / f"{config['experiment_id']}.json", config)
    store = CollectionStore(db_path)
    store.initialize_experiment(config)
    trial_id = args.trial_id or store.create_trial(
        config["experiment_id"], args.dataset_id,
        metadata={"project_uid": args.project, "workspace_uid": args.workspace},
    )
    trial = store.trial(trial_id)
    if trial["dataset_id"] != args.dataset_id:
        raise ValueError("trial dataset does not match --dataset-id")
    ctf_job_uid = getattr(args, "ctf_job_uid", None)
    if args.submit and not ctf_job_uid:
        raise ValueError("live visual RL trials require a completed Patch CTF Job (--ctf-job-uid)")
    run_dir = root / "runs" / config["experiment_id"] / trial_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "run_config.json").exists():
        write_json(run_dir / "run_config.json", {
            "experiment_id": config["experiment_id"], "trial_id": trial_id,
            "dataset_id": args.dataset_id, "project_uid": args.project,
            "workspace_uid": args.workspace, "submit": bool(args.submit),
            "ctf_job_uid": ctf_job_uid,
            "started_at": utc_now(), "data_root": str(root),
        })
    with (run_dir / "invocations.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "at": utc_now(), "submit": bool(args.submit),
            "backend": args.backend, "model": args.model if args.backend == "api" else args.base_model,
        }, ensure_ascii=False) + "\n")

    subset = trial["metadata"].get("fixed_micrograph_subset")
    if not subset:
        if not args.subset_job_uid:
            raise ValueError("first run requires --subset-job-uid for the completed Exposure Sets Tool Job")
        snapshot_args = SimpleNamespace(
            project=args.project, job_uid=args.subset_job_uid, output_name="split_0",
            output=str(run_dir / "fixed_micrograph_subset.json"),
        )
        subset_result = snapshot_exposure_subset(snapshot_args)
        subset = store.bind_micrograph_subset(trial_id, subset_result)
        write_json(run_dir / "fixed_micrograph_subset.json", subset)
    if ctf_job_uid:
        ctf_output_count = getattr(args, "ctf_output_count", None)
        ctf_dropped_uids = getattr(args, "ctf_dropped_micrograph_uids", []) or []
        subset = store.bind_micrograph_subset(
            trial_id, {
                **subset,
                "ctf_job_uid": ctf_job_uid,
                "ctf_output_name": "exposures",
                "ctf_output_count": ctf_output_count,
                "ctf_retained_micrograph_uids": getattr(args, "ctf_retained_micrograph_uids", []) or [],
                "ctf_dropped_micrograph_uids": ctf_dropped_uids,
            },
        )
        write_json(run_dir / "fixed_micrograph_subset.json", subset)
        from cryosparc_client import cryosparc_client

        ctf_job = validate_ctf_source(
            cryosparc_client().find_project(args.project), ctf_job_uid,
            args.workspace, subset["micrograph_uids"], subset["source_job_uid"],
            subset.get("ctf_output_count"),
        )
        retained_uids = [str(row["uid"]) for row in ctf_job.load_output("exposures").rows()]
        subset = store.bind_micrograph_subset(
            trial_id, {**subset, "ctf_retained_micrograph_uids": retained_uids},
        )
        write_json(run_dir / "fixed_micrograph_subset.json", subset)
    validate_micrograph_subset(subset, config["sampling"]["micrograph_count"])

    project_dir = Path(__file__).resolve().parents[1]
    server_env = os.environ.copy()
    server_env["CRYOAGENT_VISION_CACHE_DIR"] = str(root / "vision_inputs")
    server = StdioServerParameters(
        command=args.server_python,
        args=[str(project_dir / "cryosparc_agent" / "cryosparc_mcp_server.py")],
        cwd=str(project_dir), env=server_env,
    )
    stages = ("blob_picker_diameter", "inspect_picks_threshold", "extract_box_size", "class_2d")
    stage_to_visual_tool = {
        "inspect_picks_threshold": "get_pick_inspection_visual_context",
        "extract_box_size": "get_extract_box_visual_context",
    }
    dataset_infos = config.get("dataset_info", {})
    dataset_info = dataset_infos.get(args.dataset_id)
    if dataset_info is None:
        target_dataset_key = str(args.dataset_id).casefold()
        dataset_info = next(
            (value for key, value in dataset_infos.items()
             if str(key).casefold() == target_dataset_key),
            {},
        )

    model_worker_holder = getattr(args, "model_worker_holder", None)
    model_worker = model_worker_holder.get("worker") if model_worker_holder is not None else None
    if model_worker is not None:
        with (run_dir / "invocations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "at": utc_now(),
                "event": "model_worker_reused",
                "persistent_for_batch": True,
                "requests_served_before_trial": model_worker.requests_served,
            }, ensure_ascii=False) + "\n")

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            for stage in stages:
                trial = store.trial(trial_id)
                previous_job = _event_job(trial, stage)
                if previous_job and previous_job["payload"].get("status") == "completed":
                    continue

                recovered_job = None
                if not previous_job:
                    recovered_job = _execution_job_to_recover(trial, stage)
                    if recovered_job and recovered_job.get("job_uid"):
                        job_uid = str(recovered_job["job_uid"])
                        status = str(recovered_job.get("status") or "queued").lower()
                        store.record_event(trial_id, stage, "job_status", {
                            "status": status if status in {"queued", "running", "completed", "failed", "aborted"} else "queued",
                            "submitted": True,
                            "success": bool(recovered_job.get("success")),
                            "recovered_from_execution_event": True,
                        }, job_uid=job_uid)
                        trial = store.trial(trial_id)
                        previous_job = _event_job(trial, stage)

                if stage == "blob_picker_diameter":
                    current_node = subset.get("ctf_job_uid") or subset["source_job_uid"]
                    visual_tool = "get_micrograph_preview_visual_context"
                    visual_arguments = {
                        "project_uid": args.project, "job_uid": current_node,
                        "max_micrographs": args.visual_micrographs,
                        "output_name": subset.get("ctf_output_name", subset["output_name"]),
                        "micrograph_uids": subset.get("ctf_retained_micrograph_uids") or subset["micrograph_uids"],
                    }
                elif stage == "inspect_picks_threshold":
                    blob_event = _event_job(trial, "blob_picker_diameter")
                    if not blob_event or blob_event["payload"].get("status") != "completed":
                        raise RuntimeError("Inspect Picks requires the completed Blob Picker Job")
                    current_node = blob_event["job_uid"]
                    visual_tool = stage_to_visual_tool[stage]
                    visual_arguments = {"project_uid": args.project, "job_uid": current_node}
                elif stage == "extract_box_size":
                    inspect_event = _event_job(trial, "inspect_picks_threshold")
                    if not inspect_event or inspect_event["payload"].get("status") != "completed":
                        raise RuntimeError("Extract requires the completed Inspect Picks Job")
                    current_node = inspect_event["job_uid"]
                    visual_tool = stage_to_visual_tool[stage]
                    visual_arguments = {"project_uid": args.project, "job_uid": current_node}
                else:
                    extract_event = _event_job(trial, "extract_box_size")
                    if not extract_event or extract_event["payload"].get("status") != "completed":
                        raise RuntimeError("Class 2D requires the completed Extract Job")
                    current_node = extract_event["job_uid"]
                    visual_tool = None
                    visual_arguments = None

                # A recorded queued/running UID is polled on resume; it is never resubmitted.
                if previous_job:
                    job_uid = previous_job["job_uid"]
                    await _wait_for_trial_job(
                        session, store, trial_id, stage, job_uid,
                        project=args.project, workspace=args.workspace,
                        timeout_seconds=args.wait_timeout_seconds,
                        poll_interval_seconds=args.poll_interval_seconds,
                        retry_count=args.wait_retries,
                    )
                    continue

                if stage == "class_2d":
                    decision = _class2d_decision(config)
                    store.record_event(trial_id, stage, "fixed_policy_decision", {
                        "decision": decision,
                        "policy": "fixed_class2d_after_extract",
                    })
                else:
                    decision = _existing_decision(trial, stage)
                    if decision is None:
                        model_input = await call_tool_json(session, "build_model_input_payload", {
                            "project_uid": args.project,
                            "workspace_uid": args.workspace,
                            "current_job_uid": current_node,
                            "dataset_info": dataset_info,
                        })
                        if model_input.get("mcp_error") or model_input.get("ready_for_model") is False:
                            raise RuntimeError(f"Current workflow state is not decision-ready: {model_input}")
                        visual = await call_tool_json(session, visual_tool, visual_arguments)
                        if visual.get("mcp_error") or visual.get("success") is False:
                            raise RuntimeError(f"Could not build {stage} visual evidence: {visual}")
                        context = {
                            "project_uid": args.project,
                            "workspace_uid": args.workspace,
                            "current_job_uid": current_node,
                            "workflow_state": model_input,
                            "dataset_info": dataset_info,
                            "fixed_micrograph_subset": {
                                "micrograph_count": subset["micrograph_count"],
                                "fingerprint": subset["fingerprint"],
                            },
                        }
                        write_json(run_dir / f"{stage}_context.json", context)
                        write_json(run_dir / f"{stage}_visual_evidence.json", _durable_visual_context(visual))
                        decision_args = SimpleNamespace(
                            db=str(db_path), experiment_id=config["experiment_id"],
                            dataset_id=args.dataset_id, trial_id=trial_id, stage=stage,
                            context_file=None, visual_context_file=None,
                            backend=args.backend, api_base=args.api_base, api_key_env=args.api_key_env,
                            model=args.model, base_model=args.base_model, adapter=args.adapter,
                            model_python=args.model_python, model_srun_prefix=args.model_srun_prefix,
                            device_map=args.device_map, torch_dtype=args.torch_dtype,
                            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                            timeout_seconds=args.model_timeout_seconds, max_retries=args.model_retries,
                            output_dir=str(run_dir),
                        )
                        if args.backend == "local":
                            if model_worker is None:
                                from autonomous_mcp_closed_loop import ModelWorker

                                worker_args = SimpleNamespace(
                                    vision=True,
                                    model_python=args.model_python,
                                    project_dir=str(_model_worker_project_dir()),
                                    base_model=args.base_model,
                                    adapter=args.adapter,
                                    max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature,
                                    device_map=args.device_map,
                                    torch_dtype=args.torch_dtype,
                                    model_srun_prefix=args.model_srun_prefix,
                                )
                                model_worker = ModelWorker(
                                    worker_args, run_dir / "model_worker.stderr.log",
                                )
                                model_worker.start()
                                if model_worker_holder is not None:
                                    model_worker_holder["worker"] = model_worker
                                atexit.register(model_worker.close)
                                with (run_dir / "invocations.jsonl").open("a", encoding="utf-8") as stream:
                                    stream.write(json.dumps({
                                        "at": utc_now(),
                                        "event": "model_worker_loaded",
                                        "load_event": model_worker.load_event,
                                        "persistent_for_trial": True,
                                    }, ensure_ascii=False) + "\n")
                            decision_args.model_worker = model_worker
                        # The normal decision path reads JSON files; write the exact persisted context first.
                        context_file = run_dir / f"{stage}_context.json"
                        visual_file = run_dir / f"{stage}_visual_evidence.json"
                        write_json(visual_file, _durable_visual_context(visual))
                        decision_args.context_file = str(context_file)
                        decision_args.visual_context_file = str(visual_file)
                        while True:
                            invalid_before = sum(
                                event["stage"] == stage
                                and event["event_type"] == "model_decision_invalid"
                                for event in current_attempt_events(store.trial(trial_id))
                            )
                            try:
                                generated = decide_stage(decision_args)
                                decision = generated["decision"]
                                break
                            except Exception:
                                updated_trial = store.trial(trial_id)
                                invalid_after = sum(
                                    event["stage"] == stage
                                    and event["event_type"] == "model_decision_invalid"
                                    for event in current_attempt_events(updated_trial)
                                )
                                decisions_used = sum(
                                    event["stage"] == stage
                                    and event["event_type"] in {
                                        "model_decision", "model_decision_invalid",
                                    }
                                    for event in current_attempt_events(updated_trial)
                                )
                                if (
                                    invalid_after > invalid_before
                                    and decisions_used < config["limits"]["max_decisions_per_stage"]
                                ):
                                    continue
                                raise
                    else:
                        visual = {}

                execution_decision = (
                    to_v2_execution_decision(stage, decision)
                    if stage in STAGES else decision
                )
                if stage == "blob_picker_diameter":
                    execution_decision["connections"] = {
                        "micrographs": {
                            "source_job_uid": current_node,
                            "source_output": subset.get("ctf_output_name", subset["output_name"]),
                        },
                    }
                elif stage == "class_2d":
                    execution_decision["connections"] = {
                        "particles": {
                            "source_job_uid": current_node,
                            "source_output": "particles",
                        },
                    }
                submitted = await _submit_decision(
                    session, store, trial_id, stage, execution_decision,
                    project=args.project, workspace=args.workspace,
                    current_node=current_node, dataset_info=dataset_info,
                    submit=args.submit,
                )
                if not args.submit:
                    if model_worker is not None and model_worker_holder is None:
                        model_worker.close()
                    return {
                        "trial_id": trial_id, "dry_run": True,
                        "stopped_after_stage": stage, "execution": submitted,
                        "data_root": str(root),
                    }
                await _wait_for_trial_job(
                    session, store, trial_id, stage, submitted["job_uid"],
                    project=args.project, workspace=args.workspace,
                    timeout_seconds=args.wait_timeout_seconds,
                    poll_interval_seconds=args.poll_interval_seconds,
                    retry_count=args.wait_retries,
                )

            class2d_event = _event_job(store.trial(trial_id), "class_2d")
            if not class2d_event:
                raise RuntimeError("Class 2D finished without a recorded job UID")
            feedback = await call_tool_json(session, "get_class2d_feedback_context", {
                "project_uid": args.project, "job_uid": class2d_event["job_uid"],
            })
            if feedback.get("mcp_error") or feedback.get("success") is False:
                raise RuntimeError(f"Could not retrieve Class 2D feedback: {feedback}")
            try:
                projection_metrics = await call_tool_json(
                    session, "get_class2d_reference_projection_context", {
                        "project_uid": args.project,
                        "job_uid": class2d_event["job_uid"],
                        "empiar_id": dataset_info.get("empiar_id", args.dataset_id),
                        "emdb_id": dataset_info.get("emdb_id"),
                        "reference_map_path": dataset_info.get("reference_map_path"),
                    },
                )
                if projection_metrics.get("mcp_error"):
                    projection_metrics = {"status": "failed", "message": projection_metrics}
            except Exception as exc:
                projection_metrics = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}
            try:
                cryosift_result = await call_tool_json(
                    session, "evaluate_2d_classes_with_cryosift", {
                        "project_uid": args.project,
                        "job_uid": class2d_event["job_uid"],
                        "threshold": 3.0,
                    },
                )
                if cryosift_result.get("mcp_error"):
                    cryosift_result = {"success": False, "status": "tool_error", "message": cryosift_result}
            except Exception as exc:
                cryosift_result = {"success": False, "status": "failed", "message": f"{type(exc).__name__}: {exc}"}
            measured = compute_class2d_feedback(
                feedback.get("classes") or [],
                int(feedback.get("input_particle_count") or 0),
                min_usable_particles=args.min_usable_particles,
                min_particle_coverage=args.min_particle_coverage,
                baseline_resolution_A=args.baseline_resolution_A,
                reference_projection_metrics=projection_metrics,
                cryosift_result=cryosift_result,
            )
            measured["class2d_job_uid"] = class2d_event["job_uid"]
            measured["class_count"] = feedback.get("class_count")
            measured["cryosift_retention"]["raw_result_status"] = cryosift_result.get("status")
            store.record_event(trial_id, "class2d_feedback", "reference_projection_metrics", projection_metrics)
            store.record_event(trial_id, "class2d_feedback", "cryosift_result", cryosift_result)
            store.record_event(trial_id, "class2d_feedback", "measured_outcome", measured)
            store.record_outcome(trial_id, measured)
            write_json(run_dir / "class2d_feedback.json", measured)
            write_json(run_dir / "reference_projection_metrics.json", projection_metrics)
            write_json(run_dir / "cryosift_result.json", cryosift_result)
            for split in ("train", "heldout"):
                store.export_episodes(
                    config["experiment_id"],
                    root / "trajectories" / f"{config['experiment_id']}_{split}.jsonl",
                    split=split,
                )
            if model_worker is not None and model_worker_holder is None:
                model_worker.close()
                with (run_dir / "invocations.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({
                        "at": utc_now(),
                        "event": "model_worker_closed",
                        "requests_served": model_worker.requests_served,
                    }, ensure_ascii=False) + "\n")
            return {"trial_id": trial_id, "status": "completed", "feedback": measured,
                    "data_root": str(root)}


async def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Run a resumable list of 20-micrograph trials with one shared local model worker."""
    root = ensure_writable_data_root(args.data_root)
    config = validate_collection_config(read_json(args.config))
    manifest = read_json(args.batch_manifest)
    batch_id = str(manifest.get("batch_id") or "")
    rows = manifest.get("trials")
    if not batch_id or Path(batch_id).name != batch_id or batch_id in {".", ".."}:
        raise ValueError("batch manifest requires a simple batch_id without path separators")
    if not isinstance(rows, list) or not rows:
        raise ValueError("batch manifest requires a non-empty trials list")

    normalized_rows = []
    subset_jobs_by_dataset: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"trials[{index}] must be an object")
        required = ("dataset_id", "project", "workspace", "subset_job_uid")
        missing = [key for key in required if not row.get(key)]
        if args.submit and not row.get("ctf_job_uid"):
            missing.append("ctf_job_uid")
        if missing:
            raise ValueError(f"trials[{index}] is missing {missing}")
        dataset_id = str(row["dataset_id"])
        dataset_key = normalized_dataset_id(dataset_id)
        subset_job_uid = str(row["subset_job_uid"])
        prior_subset_job = subset_jobs_by_dataset.setdefault(dataset_key, subset_job_uid)
        if prior_subset_job != subset_job_uid:
            raise ValueError(
                f"dataset {dataset_id} must reuse one Exposure Sets Tool Job across batch trials"
            )
        trial_id = str(row.get("trial_id") or uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"cryoagent-visual-rl:{config['experiment_id']}:{batch_id}:{index}:{dataset_key}",
        ))
        normalized_rows.append({
            "index": index,
            "dataset_id": dataset_id,
            "project": str(row["project"]),
            "workspace": str(row["workspace"]),
            "subset_job_uid": subset_job_uid,
            "ctf_job_uid": str(row["ctf_job_uid"]) if row.get("ctf_job_uid") else None,
            "ctf_output_count": int(row["ctf_output_count"])
            if row.get("ctf_output_count") is not None else None,
            "ctf_retained_micrograph_uids": [
                str(uid) for uid in row.get("ctf_retained_micrograph_uids", [])
            ],
            "ctf_dropped_micrograph_uids": [
                str(uid) for uid in row.get("ctf_dropped_micrograph_uids", [])
            ],
            "trial_id": trial_id,
        })
    trial_ids = [row["trial_id"] for row in normalized_rows]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("each batch row must have a unique trial_id")

    digest = hashlib.sha256(json.dumps(
        {"experiment_id": config["experiment_id"], "trials": normalized_rows},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    batch_dir = root / "runs" / "batches"
    state_path = batch_dir / f"{batch_id}.json"
    if state_path.exists():
        state = read_json(state_path)
        if state.get("manifest_sha256") != digest:
            old_rows = state.get("trials") or []
            identity_keys = ("index", "dataset_id", "project", "workspace", "subset_job_uid", "trial_id")
            old_identity = [tuple(item.get(key) for key in identity_keys) for item in old_rows]
            new_identity = [tuple(item.get(key) for key in identity_keys) for item in normalized_rows]
            if old_identity != new_identity or not all(item.get("ctf_job_uid") for item in normalized_rows):
                raise ValueError("batch manifest differs from the saved batch; use a new batch_id")
            # Permit only the expected one-time upgrade from exposure-only to
            # CTF-bound trials while preserving prior trial IDs and outcomes.
            state["manifest_sha256"] = digest
            state["ctf_job_uids_added_at"] = utc_now()
            for prior, current in zip(old_rows, normalized_rows):
                prior["ctf_job_uid"] = current["ctf_job_uid"]
            write_json(state_path, state)
    else:
        state = {
            "batch_id": batch_id,
            "experiment_id": config["experiment_id"],
            "manifest_sha256": digest,
            "status": "planned",
            "trials": [
                {**row, "status": "pending", "result": None, "error": None}
                for row in normalized_rows
            ],
            "created_at": utc_now(),
        }
        write_json(state_path, state)

    if not args.submit:
        return {
            "batch_id": batch_id,
            "status": "planned",
            "trial_count": len(normalized_rows),
            "trials": normalized_rows,
            "state_file": str(state_path),
            "note": "No CryoSPARC jobs were submitted. Re-run with --submit to execute.",
        }

    store = CollectionStore(root / "collection.sqlite")
    store.initialize_experiment(config)
    for row in normalized_rows:
        store.create_trial(
            config["experiment_id"], row["dataset_id"],
            metadata={
                "project_uid": row["project"],
                "workspace_uid": row["workspace"],
                "batch_id": batch_id,
                "batch_index": row["index"],
            },
            trial_id=row["trial_id"],
        )

    state["status"] = "running"
    state["started_at"] = state.get("started_at") or utc_now()
    state["updated_at"] = utc_now()
    state["state_file"] = str(state_path)
    write_json(state_path, state)
    worker_holder: dict[str, Any] = {}
    had_trial_errors = False
    try:
        for item in state["trials"]:
            if item.get("status") == "completed":
                continue
            if item.get("status") == "failed" and getattr(args, "renew_failed_decision_budget", False):
                store.record_event(item["trial_id"], "execution", "retry_started", {
                    "previous_error": item.get("error"),
                    "previous_outcome": store.trial(item["trial_id"])["outcome"],
                    "decision_budget_renewed": True,
                    "job_budget_renewed": False,
                })
            item["status"] = "running"
            item["started_at"] = item.get("started_at") or utc_now()
            item["error"] = None
            state["updated_at"] = utc_now()
            write_json(state_path, state)

            trial_args = SimpleNamespace(**vars(args))
            trial_args.dataset_id = item["dataset_id"]
            trial_args.project = item["project"]
            trial_args.workspace = item["workspace"]
            trial_args.subset_job_uid = item["subset_job_uid"]
            trial_args.ctf_job_uid = item.get("ctf_job_uid")
            trial_args.ctf_output_count = item.get("ctf_output_count")
            trial_args.ctf_retained_micrograph_uids = item.get("ctf_retained_micrograph_uids", [])
            trial_args.ctf_dropped_micrograph_uids = item.get("ctf_dropped_micrograph_uids", [])
            trial_args.trial_id = item["trial_id"]
            trial_args.model_worker_holder = worker_holder
            try:
                result = await run_trial_with_service_retries(trial_args, state, item, state_path)
                item["result"] = result
                item["status"] = "completed" if result.get("status") == "completed" else "incomplete"
                item["finished_at"] = utc_now()
                if item["status"] != "completed":
                    raise RuntimeError(f"trial did not complete: {result}")
            except CollectionServiceUnavailable:
                # Do not advance the batch or assign a policy failure for an outage.
                raise
            except Exception as exc:
                item["status"] = "failed"
                item["error"] = {
                    "type": type(exc).__name__, "message": str(exc),
                    "leaf_errors": exception_leaves(exc),
                }
                item["finished_at"] = utc_now()
                episode = store.trial(item["trial_id"])
                if episode["outcome"] is None or (episode["outcome"].get("failure") or {}).get("type"):
                    failure = {
                        "metric_version": "class2d-feedback-0.1",
                        "episode_status": "failed",
                        "eligible_for_policy_update": False,
                        "resolution_reward": None,
                        "failure": {
                            **item["error"],
                            "stage": episode["events"][-1]["stage"] if episode["events"] else "initialization",
                        },
                    }
                    store.record_outcome(item["trial_id"], failure)
                    write_json(root / "runs" / config["experiment_id"] / item["trial_id"] / "failure_outcome.json", failure)
                for split in ("train", "heldout"):
                    store.export_episodes(
                        config["experiment_id"],
                        root / "trajectories" / f"{config['experiment_id']}_{split}.jsonl",
                        split=split,
                    )
                had_trial_errors = True
                state["status"] = "running" if args.continue_on_trial_error else "failed"
                state["updated_at"] = utc_now()
                write_json(state_path, state)
                if not args.continue_on_trial_error:
                    raise
                continue
            state["updated_at"] = utc_now()
            write_json(state_path, state)

        state["status"] = "completed_with_errors" if had_trial_errors else "completed"
        state["finished_at"] = utc_now()
        state["updated_at"] = utc_now()
        return {
            "batch_id": batch_id,
            "status": state["status"],
            "trial_count": len(normalized_rows),
            "state_file": str(state_path),
            "worker_requests_served": (
                worker_holder["worker"].requests_served if worker_holder.get("worker") else 0
            ),
            "failed_trial_count": sum(item.get("status") == "failed" for item in state["trials"]),
        }
    finally:
        worker = worker_holder.get("worker")
        if worker is not None:
            worker.close()
            state["model_worker_closed_at"] = utc_now()
            state["model_worker_requests_served"] = worker.requests_served
            state["updated_at"] = utc_now()
        write_json(state_path, state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init", help="Initialize an experiment ledger from JSON config")
    initialize.add_argument("--db", required=True)
    initialize.add_argument("--config", required=True)
    def initialize_experiment(args: argparse.Namespace) -> dict[str, Any]:
        config = read_json(args.config)
        CollectionStore(args.db).initialize_experiment(config)
        return {"initialized": True, "experiment_id": config["experiment_id"]}
    initialize.set_defaults(func=initialize_experiment)

    decide = subparsers.add_parser("decide", help="Ask the configured model for one bounded stage decision")
    decide.add_argument("--db", required=True)
    decide.add_argument("--experiment-id", required=True)
    decide.add_argument("--dataset-id", required=True)
    decide.add_argument("--trial-id")
    decide.add_argument("--stage", required=True, choices=STAGES)
    decide.add_argument("--context-file", required=True, help="JSON facts/current state for this decision")
    decide.add_argument("--visual-context-file", help="MCP visual context JSON, including attached data URLs")
    decide.add_argument("--backend", choices=("api", "local"), default="api")
    decide.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    decide.add_argument("--api-key-env", default="OPENAI_API_KEY")
    decide.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    decide.add_argument("--base-model", default="/ssd1/lisongyang/models/Qwen3.6-27B")
    decide.add_argument("--adapter")
    decide.add_argument("--model-python", default=sys.executable)
    decide.add_argument("--model-srun-prefix")
    decide.add_argument("--device-map", default="auto")
    decide.add_argument("--torch-dtype", default="bfloat16")
    decide.add_argument("--max-new-tokens", type=int, default=512)
    decide.add_argument("--temperature", type=float, default=0.2)
    decide.add_argument("--timeout-seconds", type=int, default=300)
    decide.add_argument("--max-retries", type=int, default=2)
    decide.add_argument("--output-dir", default=str(DEFAULT_DATA_ROOT / "runs" / "manual"))
    decide.set_defaults(func=lambda a: print(json.dumps(decide_stage(a), ensure_ascii=False, indent=2)))

    record = subparsers.add_parser("record-feedback", help="Record Class 2D class statistics and calculate bounded feedback")
    record.add_argument("--db", required=True)
    record.add_argument("--trial-id", required=True)
    record.add_argument("--class-statistics-file", required=True)
    record.add_argument("--input-particle-count", type=int,
                        help="Override the count from get_class2d_feedback_context output")
    record.add_argument("--min-usable-particles", type=int, default=1000)
    record.add_argument("--min-particle-coverage", type=float, default=0.5)
    record.add_argument("--baseline-resolution-A", type=float,
                        help="Matched baseline on the same dataset/subset; enables paired improvement reward")
    def record_feedback(args: argparse.Namespace) -> None:
        data = read_json(args.class_statistics_file)
        classes = data.get("classes", data) if isinstance(data, dict) else data
        input_count = args.input_particle_count
        if input_count is None and isinstance(data, dict):
            input_count = data.get("input_particle_count")
        if input_count is None:
            raise ValueError("provide --input-particle-count or use MCP class2d feedback output")
        feedback = compute_class2d_feedback(
            classes, input_count,
            min_usable_particles=args.min_usable_particles,
            min_particle_coverage=args.min_particle_coverage,
            baseline_resolution_A=args.baseline_resolution_A,
        )
        store = CollectionStore(args.db)
        store.record_event(args.trial_id, "class2d_feedback", "measured_outcome", feedback)
        store.record_outcome(args.trial_id, feedback)
        print(json.dumps({"trial_id": args.trial_id, **feedback}, ensure_ascii=False, indent=2))
    record.set_defaults(func=record_feedback)

    job = subparsers.add_parser("record-job", help="Attach a CryoSPARC Job status/result to a trial")
    job.add_argument("--db", required=True)
    job.add_argument("--trial-id", required=True)
    job.add_argument("--stage", required=True, choices=EVENT_STAGES)
    job.add_argument("--job-uid", required=True)
    job.add_argument("--status", required=True, choices=("queued", "running", "completed", "failed", "aborted"))
    job.add_argument("--result-file", help="Optional JSON with parameters and a compact job result")
    def record_job(args: argparse.Namespace) -> None:
        payload = {"status": args.status}
        if args.result_file:
            payload["result"] = read_json(args.result_file)
        CollectionStore(args.db).record_event(
            args.trial_id, args.stage, "job_status", payload, job_uid=args.job_uid,
        )
        if args.status in {"failed", "aborted"}:
            store = CollectionStore(args.db)
            store.record_outcome(args.trial_id, {
                "metric_version": "class2d-feedback-0.1",
                "episode_status": "failed",
                "eligible_for_policy_update": False,
                "resolution_reward": None,
                "failure": {"stage": args.stage, "job_uid": args.job_uid,
                            "status": args.status, "details": payload.get("result")},
                "reward_direction": "failed/aborted episode retained without a numeric resolution score",
            })
        print(json.dumps({"recorded": True, "trial_id": args.trial_id, "job_uid": args.job_uid}))
    job.set_defaults(func=record_job)

    execute = subparsers.add_parser("execute", help="Validate a saved decision; live submission requires --submit")
    execute.add_argument("--db", required=True)
    execute.add_argument("--trial-id", required=True)
    execute.add_argument("--stage", required=True, choices=STAGES)
    execute.add_argument("--project", required=True)
    execute.add_argument("--workspace", required=True)
    execute.add_argument("--current-node", required=True)
    execute.add_argument("--server-python", default=sys.executable)
    execute.add_argument("--output-dir", default=str(DEFAULT_DATA_ROOT / "runs" / "manual"))
    execute.add_argument("--submit", action="store_true",
                         help="Create/queue only after binding a fixed 20-micrograph subset")
    execute.set_defaults(func=lambda a: asyncio.run(execute_stage(a)))

    bind = subparsers.add_parser(
        "bind-subset", help="Bind a 20-micrograph Exposure Sets Tool output to a trial",
    )
    bind.add_argument("--db", required=True)
    bind.add_argument("--trial-id", required=True)
    bind.add_argument(
        "--subset-file", required=True,
        help="JSON containing source_job_uid, output_name, and 20 micrograph_uids",
    )
    def bind_subset(args: argparse.Namespace) -> None:
        checked = CollectionStore(args.db).bind_micrograph_subset(
            args.trial_id, read_json(args.subset_file),
        )
        print(json.dumps({"trial_id": args.trial_id, "fixed_micrograph_subset": checked},
                         ensure_ascii=False, indent=2))
    bind.set_defaults(func=bind_subset)

    snapshot = subparsers.add_parser(
        "snapshot-subset", help="Read and verify the 20 exposure UIDs from CryoSPARC",
    )
    snapshot.add_argument("--project", required=True)
    snapshot.add_argument("--job-uid", required=True, help="Completed Exposure Sets Tool Job")
    snapshot.add_argument("--output-name", default="split_0")
    snapshot.add_argument("--output", required=True, help="JSON path for bind-subset")
    snapshot.set_defaults(func=snapshot_exposure_subset)

    export = subparsers.add_parser("export", help="Export completed trials as trajectory JSONL")
    export.add_argument("--db", required=True)
    export.add_argument("--experiment-id", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--split", choices=("train", "heldout"), default="train")
    export.set_defaults(func=lambda a: print(json.dumps({
        "episodes_exported": CollectionStore(a.db).export_episodes(a.experiment_id, a.output, a.split),
        "split": a.split,
        "output": a.output,
    }, ensure_ascii=False)))

    run = subparsers.add_parser(
        "run-trial",
        help="Run/resume the complete 20-micrograph visual decision -> Class 2D loop",
    )
    run.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                     help="All DB, logs, image cache, and trajectories stay under this writable path")
    run.add_argument("--config", required=True, help="Validated experiment config JSON")
    run.add_argument("--dataset-id", required=True)
    run.add_argument("--trial-id", help="Resume an existing trial instead of creating one")
    run.add_argument("--subset-job-uid", help="Completed Exposure Sets Tool Job (split_0, exactly 20 UIDs)")
    run.add_argument("--ctf-job-uid", help="Completed Patch CTF Job for the same 20-micrograph subset")
    run.add_argument("--project", required=True)
    run.add_argument("--workspace", required=True)
    run.add_argument("--server-python", default=sys.executable)
    run.add_argument("--submit", action="store_true", help="Actually create/queue CryoSPARC Jobs")
    run.add_argument("--backend", choices=("api", "local"), default="api")
    run.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    run.add_argument("--base-model", default="/ssd1/lisongyang/models/Qwen3.6-27B")
    run.add_argument("--adapter")
    run.add_argument("--model-python", default=sys.executable)
    run.add_argument("--model-srun-prefix")
    run.add_argument("--device-map", default="auto")
    run.add_argument("--torch-dtype", default="bfloat16")
    run.add_argument("--max-new-tokens", type=int, default=512)
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--model-timeout-seconds", type=int, default=300)
    run.add_argument("--model-retries", type=int, default=2)
    run.add_argument("--visual-micrographs", type=int, default=6)
    run.add_argument("--wait-timeout-seconds", type=int, default=43200)
    run.add_argument("--poll-interval-seconds", type=int, default=45)
    run.add_argument("--wait-retries", type=int, default=3)
    run.add_argument("--min-usable-particles", type=int, default=1000)
    run.add_argument("--min-particle-coverage", type=float, default=0.5)
    run.add_argument("--baseline-resolution-A", type=float)
    run.set_defaults(func=lambda a: asyncio.run(run_trial(a)))

    batch = subparsers.add_parser(
        "run-batch",
        help="Run/resume sequential 20-micrograph trials, reusing one local model worker",
    )
    batch.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    batch.add_argument("--config", required=True, help="Validated experiment config JSON")
    batch.add_argument("--batch-manifest", required=True, help="JSON manifest listing trial workspaces/subset Jobs")
    batch.add_argument("--server-python", default=sys.executable)
    batch.add_argument("--submit", action="store_true", help="Actually execute CryoSPARC Jobs; omit for plan-only")
    batch.add_argument("--backend", choices=("api", "local"), default="api")
    batch.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    batch.add_argument("--api-key-env", default="OPENAI_API_KEY")
    batch.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    batch.add_argument("--base-model", default="/ssd1/lisongyang/models/Qwen3.6-27B")
    batch.add_argument("--adapter")
    batch.add_argument("--model-python", default=sys.executable)
    batch.add_argument("--model-srun-prefix")
    batch.add_argument("--device-map", default="auto")
    batch.add_argument("--torch-dtype", default="bfloat16")
    batch.add_argument("--max-new-tokens", type=int, default=512)
    batch.add_argument("--temperature", type=float, default=0.2)
    batch.add_argument("--model-timeout-seconds", type=int, default=300)
    batch.add_argument("--model-retries", type=int, default=2)
    batch.add_argument("--visual-micrographs", type=int, default=6)
    batch.add_argument("--wait-timeout-seconds", type=int, default=43200)
    batch.add_argument("--poll-interval-seconds", type=int, default=45)
    batch.add_argument("--wait-retries", type=int, default=3)
    batch.add_argument("--min-usable-particles", type=int, default=1000)
    batch.add_argument("--min-particle-coverage", type=float, default=0.5)
    batch.add_argument("--baseline-resolution-A", type=float)
    batch.add_argument("--service-retry-delay-seconds", type=float, default=30)
    batch.add_argument("--service-retry-max-delay-seconds", type=float, default=300)
    batch.add_argument("--service-retry-max-wait-seconds", type=float, default=21600,
                       help="Maximum service outage window before persisting a pending checkpoint and exiting")
    batch.add_argument(
        "--renew-failed-decision-budget", action="store_true",
        help="On explicit resume, grant failed trials one fresh bounded decision attempt; retain audit history",
    )
    batch.add_argument(
        "--continue-on-trial-error",
        action="store_true",
        help="Record a failed dataset trial and continue with later trials in the batch",
    )
    batch.set_defaults(func=lambda a: asyncio.run(run_batch(a)))

    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
