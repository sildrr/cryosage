from __future__ import annotations

import glob
import json
import math
import os
import re
import shlex
import subprocess
import uuid
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("CRYOAGENT_WEB_DATA_ROOT", str(Path.home() / '.local/share/cryosage')))
PYTHON = os.getenv("CRYOAGENT_WEB_PYTHON", sys.executable)
LOGS = Path(os.getenv("CRYOAGENT_WEB_LOG_ROOT", str(REPO / "logs/web")))
SHOWCASE = [
    {"id": "10025", "name": "20S proteasome", "resolution": 2.6997,
     "project": "P6", "workspace": "W21", "job": "J370", "symmetry": "D7",
     "description": "Improved from 3.52 A to 2.70 A through 2D selection and template re-picking.",
     "source": "ppt_group_meeting_20260916/build_w21_iterative_optimization_slide.py", "metric": "Refine FSC"},
    {"id": "11280", "name": "EMPIAR-11280 reconstruction", "resolution": 4.296,
     "project": "P6", "workspace": "W8", "job": "J117", "symmetry": None,
     "description": "Knowledge-assisted reconstruction; J118 validated approximately 4.30 A, with CryoNet handoff tested.",
     "source": "CRYONET_API_TEST_INDEX.md", "metric": "J118 validation FSC"},
    {"id": "10208", "name": "EMPIAR-10208 reconstruction", "resolution": 3.793032172689106,
     "project": "P18", "workspace": "W12", "job": "J282", "symmetry": "C1",
     "description": "Reconstructed from image import through Homogeneous Refinement, with final FSC resolution 3.79 A.",
     "source": "CryoSPARC P18/W12/J282; fsc_info.radwn_final_A; import J203: /home/share/empiar/10208/data/10mM_ATP_DW_micrographs/*.mrc; verified 2026-10-15",
     "metric": "Final refine FSC"},
    {"id": "10012", "name": "EMPIAR-10012 reconstruction", "resolution": 3.1900709437221684,
     "project": "P18", "workspace": "W13", "job": "J306", "symmetry": "C1",
     "description": "Reconstructed from image import through Homogeneous Refinement, with final FSC resolution 3.19 A.",
     "source": "CryoSPARC P18/W13/J306; fsc_info.radwn_final_A; import J297: /home/share/empiar/10012/data/*.mrc; verified 2026-10-15",
     "metric": "Final refine FSC"},
]


def now():
    return datetime.now(timezone.utc).isoformat()


def api_models():
    return {
        "planner": os.getenv("OPENAI_MODEL", "openai/gpt-6.1-sol"),
        "execution": os.getenv("CRYOAGENT_FIXED_API_MODEL", "openai/gpt-6-luna"),
    }


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temp.replace(path)


def job_path(identifier):
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise ValueError("Invalid task identifier")
    return DATA / "tasks" / identifier


def read_task(identifier):
    return json.loads((job_path(identifier) / "task.json").read_text())


@lru_cache(maxsize=8)
def trusted_input_roots(configured_roots, upload_root):
    roots = [Path(x).resolve() for x in configured_roots.split(":") if x]
    approved = list(roots)
    # The shared EMPIAR catalog exposes mounted datasets through numeric directory
    # links. Trust those catalog entries, not arbitrary nested file symlinks.
    for root in roots:
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            if re.fullmatch(r"\d{5,6}m?", entry.name) and entry.is_symlink() and entry.is_dir():
                approved.append(entry.resolve())
    approved.append(Path(upload_root).resolve())
    return tuple(approved)


def allowed_path(path):
    roots = trusted_input_roots(
        os.getenv("CRYOAGENT_WEB_INPUT_ROOTS", "/home/share/empiar"), str(DATA / "uploads"),
    )
    actual = Path(path).resolve()
    return any(actual.is_relative_to(root) for root in roots)


def particle_input_files(source, star):
    """Validate particle paths before expansion and every resolved stack target."""
    for value in (source, star):
        if not isinstance(value, str) or not value.startswith("/") or any(c in value for c in ("\x00", "\n", "\r")):
            raise ValueError("Particle inputs require absolute server paths")
    if glob.has_magic(star) or "$" in star or Path(star).suffix.lower() != ".star":
        raise ValueError("particle_star_path must name one STAR file (no glob)")
    if not allowed_path(star) or not Path(star).is_file() or not os.access(star, os.R_OK):
        raise ValueError("STAR file is unreadable or outside configured input roots")
    prefix = re.split(r"[?*\[]", source, maxsplit=1)[0]
    check = prefix if prefix.endswith("/") or not glob.has_magic(source) else str(Path(prefix).parent)
    if not allowed_path(check):
        raise ValueError("Input path is outside configured input roots")
    if Path(source).is_dir():
        files = sorted(str(p) for p in Path(source).iterdir() if p.suffix.lower() in {".mrc", ".mrcs"})
    else:
        files = sorted(glob.glob(source))
    if not files or any(not allowed_path(f) or not Path(f).is_file() or not os.access(f, os.R_OK) for f in files):
        raise ValueError("No readable stacks, or an input resolves outside allowed roots")
    if any(Path(f).suffix.lower() not in {".mrc", ".mrcs"} for f in files):
        raise ValueError("Particle stacks must be MRC or MRCS files")
    if len({Path(f).name for f in files}) != len(files):
        raise ValueError("Particle stack basenames must be unique")
    validate_particle_star(star, files)
    return files


def validate_particle_star(star, files):
    """Check image references without loading the particle table into memory."""
    available = {Path(f).name: Path(f).resolve() for f in files}
    references = {}
    labels = []
    in_loop = False
    column = 0
    rows = 0
    with Path(star).open() as stream:
        tokens = shlex.shlex(stream, posix=True)
        tokens.whitespace_split = True
        for token in tokens:
            if token == "loop_" or token.startswith("data_") or token == "stop_":
                if column:
                    raise ValueError("Incomplete STAR loop row")
                in_loop = token == "loop_"
                labels = []
                continue
            if not in_loop:
                continue
            if token.startswith("_") and column == 0:
                labels.append(token)
                continue
            if not labels:
                raise ValueError("Invalid STAR loop headers")
            if labels[column] == "_rlnImageName":
                index, separator, reference = token.partition("@")
                path = Path(reference)
                if not separator or not index.isdigit() or int(index) < 1 or not reference:
                    raise ValueError("STAR rlnImageName must be a positive index@stack path")
                if ".." in path.parts or "$" in reference or any(c in reference for c in ("\x00", "\n", "\r", "\\")):
                    raise ValueError("STAR stack references must not contain traversal or environment variables")
                if path.name not in available:
                    raise ValueError(f"STAR references a stack not selected by input_path: {reference}")
                if path.is_absolute() and (not allowed_path(path) or path.resolve() != available[path.name]):
                    raise ValueError("Absolute STAR references must resolve to the selected allowed stack; use relative paths for relocated data")
                previous = references.setdefault(path.name, reference)
                if previous != reference:
                    raise ValueError(f"Ambiguous STAR stack basename: {path.name}")
                rows += 1
            column = (column + 1) % len(labels)
    if column or not rows:
        raise ValueError("STAR must contain a complete particle loop with _rlnImageName")


def validate_request(payload):
    if payload.get("external_consent") is not True:
        raise ValueError("Please authorize sending decision context and evidence to the configured model API")
    source = str(payload.get("input_path", ""))
    kind = payload.get("input_type", "micrographs")
    if kind not in {"micrographs", "movies", "particles"}:
        raise ValueError("Input type must be micrographs, movies or particles")
    if kind == "particles":
        files = particle_input_files(source, payload.get("particle_star_path", ""))
        result = {"input_path": source, "input_file_count": len(files),
                  "particle_star_path": payload["particle_star_path"]}
    else:
        result = validate_image_input(source)
    gain = payload.get('gainref_path')
    if gain not in (None, ''):
        if kind != 'movies' or not isinstance(gain, str) or not gain.startswith('/'):
            raise ValueError('Gain reference requires a movie input and an absolute file path')
        if (glob.has_magic(gain) or any(c in gain for c in ('\x00','\n','\r'))
                or not allowed_path(gain) or not Path(gain).is_file() or not os.access(gain,os.R_OK)
                or Path(gain).suffix.lower() not in {'.mrc','.tif','.tiff'}):
            raise ValueError('Gain reference must be one readable MRC/TIFF inside configured input roots')
        result['gainref_path'] = gain
    return validate_parameters(payload, result, kind)


def validate_image_input(source):
    if not source.startswith("/") or any(x in source for x in ("\x00", "\n", "\r")):
        raise ValueError("Provide an absolute server path or upload files")
    # Verify the static prefix before globbing, then verify every matched symlink target.
    prefix = re.split(r"[?*\[]", source, maxsplit=1)[0]
    if not allowed_path(prefix if prefix.endswith("/") else str(Path(prefix).parent)):
        raise ValueError("Input path is outside configured input roots")
    files = sorted(glob.glob(source))
    if not files or any(not Path(f).is_file() or not allowed_path(f) for f in files):
        raise ValueError("No readable files, or an input resolves outside allowed roots")
    if any(Path(f).suffix.lower() not in {".mrc", ".mrcs", ".tif", ".tiff", ".eer"} for f in files):
        raise ValueError("Supported inputs: MRC, MRCS, TIFF and EER")
    return {"input_path": source, "input_file_count": len(files)}


def validate_parameters(payload, result, kind):
    for field, low, high in (("pixel_size_A", 0.01, 20), ("voltage_kV", 50, 400),
                             ("cs_mm", 0, 10), ("dose", 0.01, 1000), ("target", 0.5, 30)):
        if kind == "particles" and field == "dose" and payload.get(field) in (None, ""):
            result[field] = None
            continue
        value = float(payload.get(field, "nan"))
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{field} must be in [{low}, {high}]")
        result[field] = value
    result.update(input_type=kind, name=str(payload.get("name", "Untitled dataset"))[:100],
                  empiar_id=str(payload.get("empiar_id", ""))[:40], kb=bool(payload.get("kb", True)))
    symmetry = payload.get("symmetry")
    symmetry = str(symmetry).strip().upper() if symmetry is not None else ""
    if symmetry and not re.fullmatch(r"(?:C[1-9][0-9]*|D[2-9][0-9]*|T|O|I)", symmetry):
        raise ValueError("Unsupported point-group symmetry")
    rounds = int(payload.get("max_jobs", 40))
    if not 5 <= rounds <= 80:
        raise ValueError("Execution-round budget must be between 5 and 80")
    result.update(symmetry=symmetry or None, max_jobs=rounds, external_consent=True)
    from webapp.benchmark import validate_profile
    result = validate_profile(payload, result)
    return result


def controller_command(identifier):
    command = ["sbatch", "--parsable", "--output", str(LOGS / f"{identifier}-%j.log"),
               "--error", str(LOGS / f"{identifier}-%j.log")]
    partition = os.getenv("CRYOAGENT_WEB_SLURM_PARTITION", "").strip()
    if partition:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", partition):
            raise ValueError("Invalid controller Slurm partition")
        command.append(f"--partition={partition}")
    nodelist = os.getenv('CRYOAGENT_WEB_SLURM_NODELIST', '').strip()
    if nodelist:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*', nodelist):
            raise ValueError('Invalid controller Slurm node list')
        command.append(f'--nodelist={nodelist}')
    qos = os.getenv("CRYOAGENT_WEB_SLURM_QOS", "").strip()
    if qos:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", qos):
            raise ValueError("Invalid controller Slurm QoS")
        command.append(f"--qos={qos}")
    script = Path(os.getenv("CRYOAGENT_WEB_SLURM_SCRIPT", str(REPO / "webapp/controller.sbatch"))).expanduser().resolve()
    if not script.is_file():
        raise ValueError("Controller Slurm script does not exist")
    return command + [str(script), identifier]


def submit_task(payload):
    config = validate_request(payload)
    from sys import path
    path.insert(0, str(REPO / "cryosparc_agent"))
    from model_direct_runner import resolve_api_key
    resolve_api_key()  # Fail before scheduling if the server has no model credential.
    identifier = uuid.uuid4().hex
    root = job_path(identifier)
    task = {"id": identifier, "created_at": now(), "status": "submitting", "config": config,
            "model_config": api_models()}
    from webapp.pilot_pipeline import pin
    pin(task, payload)
    write(root / "task.json", task)
    if task.get('pipeline_version'):
        import hashlib
        write(root/'strategy_sources.json', dict(pipeline_version=task['pipeline_version'],
            recorded_at=now(), sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in ('cryosparc_agent', 'webapp') for p in (REPO/folder).glob('*.py')}))
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        command = controller_command(identifier)
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        task = read_task(identifier)
        task.update(slurm_job=result.stdout.strip().split(";")[0])
        if task["status"] == "submitting":
            task["status"] = "queued"
    except Exception:
        task.update(status="submission_failed")
        write(root / "task.json", task)
        raise
    write(root / "task.json", task)
    return task


def public_task(identifier):
    from webapp.usage import task_usage
    task = read_task(identifier)
    root = job_path(identifier)
    from webapp.pilot_approval import candidates
    approvals = candidates(root)
    task['pilot_diversity_override'] = dict(available=bool(approvals), best_route=approvals[0] if approvals else None)
    task["usage"] = task_usage(root)
    for name in ("progress", "result", "cryonet", "pilot_progress"):
        p = root / f"{name}.json"
        if p.exists():
            task[name] = json.loads(p.read_text())
    compatibility_snapshot = root / "progress_compat.json"
    if compatibility_snapshot.exists():
        task["progress"] = json.loads(compatibility_snapshot.read_text())
    control = root / "control/state.json"
    if control.exists() and task["status"] == "running" and json.loads(control.read_text()).get("paused"):
        task["status"] = "paused"
    if task["status"] == "finished" and (
        task.get("stop_reason") == "model_call_error" or task.get("result", {}).get("status") != "available"
    ):
        task["status"] = "needs_attention"
        task["error"] = f"Agent stopped: {task.get('stop_reason', 'no_refine_result')}; workflow did not complete"
    submission = root / "cryonet_submission/submission.json"
    if submission.exists() and task.get("cryonet", {}).get("status") == "submitting":
        response = json.loads(submission.read_text()).get("response", {})
        task["cryonet"] = {"status": "submitted", "jobid": response.get("jobid"), "url": "https://cryonet.ai"}
    supervision = root / 'service_supervision/status.json'
    if supervision.exists():
        service = json.loads(supervision.read_text())
        task['service_supervision'] = service
        if service.get('status') in {'waiting_for_cryosparc', 'waiting_for_model_api_network', 'service_stabilizing'}:
            # Keep live-control/delete protections for a controller waiting on services.
            task['status'] = 'running'
            task['controls_supported'] = True
            task.pop('error', None)
            if service.get('controller'):
                task['slurm_job'] = service['controller']
    # Query the controller scheduler, so a killed/pending controller is visible too.
    if task.get("slurm_job") and task["status"] in {"queued", "running"}:
        try:
            q = subprocess.run(["sacct", "-j", task["slurm_job"], "--noheader", "--parsable2",
                                "--format=JobID,State,ExitCode"], capture_output=True, text=True, timeout=5)
            lines = [line.split("|") for line in q.stdout.splitlines()]
            row = next((x for x in lines if x[0] == task["slurm_job"]), None)
            if row:
                task["scheduler_status"] = row[1]
                if row[1].split()[0] in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
                    task["status"] = "controller_failed"
        except (OSError, subprocess.TimeoutExpired):
            task["scheduler_status"] = "unavailable"
    return task
