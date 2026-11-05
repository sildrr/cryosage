# Run an autonomous model -> MCP -> CryoSPARC closed-loop test.
import argparse
import asyncio
import copy
import json
import os
import shlex
import subprocess
import sys
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_direct_runner import (
    parse_model_decision_text,
    resolve_api_key,
    run_openai_compatible_model,
)
from visual_decision_contract import (
    EXTRACT_BOX_PREFERENCE_GUIDANCE,
    REFINEMENT_PREFERENCE_GUIDANCE,
    STAGE_ACTIONS,
    build_stage_decision_messages,
)
from kb_tool_loop import run_kb_tool_call_loop
from selection_safety import SELECTION_SAFETY_GUIDANCE, guard_select_decision
from symmetry_policy import read_memory, remember_evidence
from stage_guidance import (
    DEFAULT_CARDS_PATH, SupplementalGuidance, last_completed_from_model_input,
)
from experiment_planner import (
    _input_signature,
    apply_anti_refinement_loop,
    build_planning_prompt,
    build_policy_plan,
    decision_respects_plan,
    restrict_candidate_context,
    validate_plan,
)
from volume_scale import estimate_template_picker_volume_scale
from workflow_completion import resolve_completion, validation_decision, completion_check
from extract_review import review_extraction, review_key, is_full_extraction, historical_extract_results, REVIEW_INSTRUCTIONS, ReviewBlocked
from cryosift_adapter import (
    CRYOSIFT_SCORE_GUIDANCE,
    CRYOSIFT_EMPTY_GUIDANCE,
    load_cached_cryosift_result,
)


DEFAULT_BASE_MODEL = "/ssd1/lisongyang/models/Qwen3.6-27B-ms-test"
DEFAULT_ADAPTER = "/ssd1/lisongyang/outputs/cryoagent-fsdp-lora-h20-v2-no-workflow"
DEFAULT_MODEL_PYTHON = os.getenv("CRYOAGENT_MODEL_PYTHON", sys.executable)
DEFAULT_SERVER_PYTHON = os.getenv("CRYOAGENT_MCP_SERVER_PYTHON", sys.executable)
DEFAULT_PROJECT_DIR = str(Path(__file__).resolve().parent)
DEFAULT_MCP_SERVER = "cryosparc_mcp_server.py"

SYSTEM_PROMPT = (
    "You are the only decision maker for an autonomous CryoSPARC workflow test. "
    "Return exactly one JSON object. Do not include markdown, comments, or "
    "thinking text. Codex and MCP will not repair missing decisions for you."
)


def rollback_class2d_job_uids(candidate_actions: list[dict[str, Any]]) -> list[str]:
    """Return distinct Class 2D sources required by visual Select 2D rollbacks.

    Older quality rollbacks use ``quality_rollback_branch`` while the bounded
    refinement menu uses ``rollback_source``.  Both describe the same evidence
    dependency and must be accepted so a safe re-selection never loses its
    required contact sheet.
    """
    job_uids: list[str] = []
    for action in candidate_actions:
        if not action.get("quality_rollback"):
            continue
        source = (
            action.get("quality_rollback_branch")
            or action.get("rollback_source")
            or {}
        )
        job_uid = source.get("class2d_job_uid")
        if isinstance(job_uid, str) and job_uid and job_uid not in job_uids:
            job_uids.append(job_uid)
    return job_uids


def attach_template_picker_volume_scale_evidence(
    candidate_context: dict[str, Any], project_uid: str,
) -> list[dict[str, Any]]:
    """Attach mask-derived diameter evidence to Template Picker candidates.

    The adapter merges ``default_parameters`` into the executed decision, so a
    reliable refine-mask measurement is used even when the model does not echo
    the diameter explicitly.
    """
    evidence_items = []
    for candidate in candidate_context.get("candidate_actions") or []:
        if candidate.get("job_type") != "template_picker_gpu":
            continue
        templates = (candidate.get("required_inputs") or {}).get("templates") or []
        template_job_uid = next(
            (item.get("source_job_uid") for item in templates if item.get("source_job_uid")),
            None,
        )
        if not template_job_uid:
            continue
        evidence = estimate_template_picker_volume_scale(project_uid, str(template_job_uid))
        candidate["volume_scale_evidence"] = evidence
        evidence_items.append(evidence)
        if not evidence.get("success"):
            continue
        diameter_A = evidence["recommended_particle_diameter_A"]
        defaults = dict(candidate.get("default_parameters") or {})
        defaults["diameter"] = diameter_A
        candidate["default_parameters"] = defaults
        parameter_template = dict(candidate.get("parameter_template") or {})
        diameter_spec = dict(parameter_template.get("diameter") or {"type": "number", "minimum": 0})
        diameter_spec["default"] = diameter_A
        diameter_spec["description"] = (
            "Mask-derived particle diameter from source refinement "
            f"{evidence['source_refine_job_uid']}: {diameter_A} A. Override only with direct picking evidence."
        )
        parameter_template["diameter"] = diameter_spec
        candidate["parameter_template"] = parameter_template
        candidate["description"] = (
            f"{candidate.get('description', '')} Source refinement mask estimates a "
            f"{diameter_A} A particle diameter; use this default unless direct evidence contradicts it."
        ).strip()
    return evidence_items

STATIC_DECISION_INSTRUCTIONS = {
    "instruction": (
        "Choose exactly one next action, rollback, request_input, or stop from the current "
        "CryoSPARC state. Only choose a job type present in candidate_actions_from_mcp. "
        "If you choose a job, include every required input connection you want MCP to use. "
        "MCP will return validation or execution errors without repairing your decision. "
        "Before deciding, you must call at least one read-only kb_* tool to retrieve relevant similar cases, official job documentation, next-step statistics, or failure evidence. "
        "Use KB evidence as advice only: the live candidate_actions_from_mcp list and current state are authoritative, and never invent a Job ID or input path. "
        "Decision-type rule: use decision_type=forward whenever you submit one ordinary candidate Job, even if you selected that Job by comparing multiple completed branches. "
        "In particular, selecting the best Class 2D branch and submitting select_2D is a forward decision, not a branch decision. "
        "Use decision_type=branch only when the selected candidate action_type is explicitly branch or when submitting multiple independent exploratory actions. "
        "When particle diameter is missing, do not request_input solely for that reason: prefer a forward Blob Picker with a deliberately broad exploratory diameter range, and explain the uncertainty in reason/risk_flags. "
        "Use request_input only when no scientifically reasonable exploratory range can be proposed or another required input is unavailable. "
        "When symmetry_evidence from ProSHADE is present, treat it as quantitative advice, not "
        "ground truth. If proposing a non-C1 refinement, preserve the C1 path and compare an "
        "evidence-backed symmetry-constrained branch rather than silently overwriting it. "
        "When dataset_info.symmetry_status is unknown, C1 is an unconstrained starting "
        "hypothesis, not known molecular symmetry. Start conservatively with C1 unless "
        "strong evidence supports another point group. You may change abinit_symmetry or "
        "refine_symmetry in an allowed candidate based on map/ProSHADE evidence, explain "
        "the evidence and uncertainty, preserve the original result, and compare corrected "
        "FSC and structural plausibility. Missing or failed ProSHADE evidence does not prove C1. "
        "For Inspect Picks, use raw pick_stats scores with calibrate_ncc=false and calibrate_pow=false. "
        "Heatmap brightness means count density, not high scores or verified quality. "
        "Preserve plausible dense particle populations unless visual evidence "
        "indicates that a region is an artifact. An explicit finite lpower_thresh_max is mandatory. "
        "Prefer a more aggressive active upper cutoff preserving most of the main bright density core "
        "while trimming sparse high-Power outliers; do not simply place it above the observed maximum. "
        "Use raw-score evidence and explicit thresholds rather than auto clustering."
        " Inspect Picks requires at least one actual NCC, local-power, curvature, or sinuosity "
        "score threshold; automatic clustering and keep_threshold alone are invalid for this "
        "autonomous high-resolution workflow."
        " At a below-target refinement checkpoint, candidate_actions_from_mcp is a bounded "
        "menu of alternative experiments. Choose exactly one. Do not repeat unchanged refine. "
        + CRYOSIFT_SCORE_GUIDANCE + " " + CRYOSIFT_EMPTY_GUIDANCE + " "
        + SELECTION_SAFETY_GUIDANCE
        + REFINEMENT_PREFERENCE_GUIDANCE
        + "When those preferred options are unavailable or contradicted by evidence, consider "
        "reclassifying selected particles to address residual heterogeneity. "
        "If the result is poor, first consider re-selecting the corresponding "
        "Class 2D result: retain clear signal classes but remove the highest-score CryoSift classes "
        "that still show weak signal. Use a new box-size branch when the class averages indicate "
        "a scale or clipping problem; use Inspect Picks or Blob Picker rollback only when their "
        "score/pick evidence indicates junk contamination or an implausible particle diameter. "
        "An alternate refinement strategy is a one-time comparison, not permission to repeat "
        "the same refinement type on unchanged particles. Prefer upstream particle-quality fixes "
        "when the map shape or Class 2D evidence is poor. "
        "For every reselect_2d rollback, obey reselection_constraint: choose a nonempty strict "
        "subset of the previous_selected_templates. Remove one or more lower-quality selected "
        "classes; never retain the same count and never add a previously excluded class. "
        "When Ab-initio three-view evidence is available, compare each view with the selected "
        "Class 2D population. If one Ab-initio view has no corresponding selected Class 2D "
        "orientation, treat that Ab-initio result as failed and do not repeat unchanged refine. "
        "On the first Ab-initio/Class 2D view mismatch, prioritize reselect_2d from the existing "
        "Class 2D run: exclude clearly incomplete, off-center, or density-shifted classes rather "
        "than running another Class 2D pass that can reproduce the same bad classes. If this "
        "mismatch recurs across two or more reconstruction attempts from the same extraction "
        "scale, treat new_box_size with a larger extraction box as the leading hypothesis, while "
        "still choosing another rollback when its direct evidence is stronger. "
        + EXTRACT_BOX_PREFERENCE_GUIDANCE
        + "When scale_observer_evidence is attached, cross-check its particle_diameter_range_A "
        "and recommended_box_size_pix against the full-field evidence and the 2.5D empirical bound. "
        "The estimate uses the "
        "particle-count-weighted 5% and 10% radial-profile diameters, the peak of the CTF-fit "
        "resolution distribution as d, and median-plus-one-sigma quality-filtered absolute defocus "
        "as deltaF. Do not substitute the target/final FSC resolution for d. CTF evidence should come "
        "from the complete Patch CTF exposure set, not a sampled pilot subset. It is an evidence-backed "
        "default, not a guarantee of optimality. "
        "At the start of a new workflow, MCP may constrain Blob Picker and Extract to a small "
        "two-pass scale-calibration pilot after Patch CTF. Follow the offered calibration actions; "
        "the first pilot extraction starts at 384 px; 720 px is only a hard upper bound, not a default. "
        "Never reconstruct pilot particles directly. For uncertain-diameter trials, an offered "
        "Select 2D may select relatively better class averages solely as full-data Template Picker "
        "templates, not as reconstruction particles. After both "
        "Class 2D measurements, compare the second-pass diameter and CTF-aware box defaults with "
        "the full-field diameter-based box preference before choosing the full-data parameters. "
        "For the full-data picker, set CryoSPARC's "
        "num_process parameter to the complete upstream micrograph count; its default of 1 processes "
        "only one micrograph. "
        "Never propose box_size_pix above 720. At the 720-pixel cap, do not expand extraction "
        "again; prioritize particle-selection or upstream picking alternatives and flag possible truncation risk. "
        "Choose template_repick only when the current resolution is within 0.5 A of the target, "
        "the Ab-initio/Class 2D template-matching min_best_ncc is at least 0.75, and the volume "
        "has a plausible, correct overall shape. Otherwise do not create templates "
        "from it because an incorrect reference can reinforce wrong picks. When eligible, prefer it "
        "as described above, especially if the existing picking population appears "
        "incomplete or contaminated; this creates templates from the "
        "volume and then runs the fixed Template Picker funnel. After Select 2D, use the same "
        "source volume for refinement rather than re-running Ab-initio. "
        "This Create Template eligibility gate applies only when deciding whether to create new "
        "templates from a refinement volume. Once a completed Create Templates job is the current "
        "node and Template Picker is an available candidate, the source volume was already chosen; "
        "do not block Template Picker merely because that intermediate job itself has no FSC or "
        "Ab-initio/Class 2D matching fields. Proceed with the provided templates and available "
        "micrographs. "
        "For Template Picker, normally leave min_distance unspecified and retain the CryoSPARC "
        "default. Change it only when direct picking evidence justifies doing so; if changed, it "
        "is a unitless multiplier of particle diameter, never an Angstrom value. "
        "For Extract From Micrographs (GPU), do not use Fourier-crop/downsampling for a high-"
        "resolution reconstruction; preserve native sampling. bin_size_pix is the absolute "
        "Fourier-cropped OUTPUT EDGE LENGTH IN PIXELS, not a downsampling factor: "
        "box_size_pix=384 with bin_size_pix=2 gives 2x2 images, not two-fold binning. "
        "Prefer leaving it unset/blank, or equal to box_size_pix when an explicit output "
        "size is required. Never set 1/2 merely to satisfy a format check or mean no binning. "
        "During resolution optimization avoid downsampling: it increases output pixel size "
        "and can discard high-frequency information needed for the target resolution. "
        "For single-class Ab-initio, "
        "a blank particle-count parameter can still make CryoSPARC use only a subset; do not "
        "assume it used every Select 2D particle. When the Ab-initio output omits many selected "
        "particles, compare the two baseline Homo-refine candidates: using its class-0 subset "
        "versus using the complete upstream Select 2D particle stack with the same Ab-initio "
        "volume. Choose based on particle quality and coverage rather than always choosing "
        "the larger stack. Consider multiple Ab-initio classes when heterogeneity is plausible."
        + (
            " When quality_rollback guidance is present and the preferred one-time NU-refine "
            "or eligible template repicking is unavailable or unsupported by evidence, consider "
            "its Select 2D recovery candidate instead of repeating unchanged refinement."
            if os.getenv("CRYOAGENT_ENABLE_QUALITY_ROLLBACK", "0").lower()
            in {"1", "true", "yes", "on"}
            else ""
        )
    ),
    "output_contract": {
        "schema_version": "2.0",
        "decision_type": "forward | branch | rollback | stop | request_input",
        "action": "CryoSPARC job type for forward/branch decisions.",
        "job_type": "Same as action when using compact format.",
        "action_id": "Required in planning mode; copy exactly from the approved candidate action.",
        "parameters": "Only non-default parameters explicitly chosen by the model.",
        "connections": {
            "input_name": {
                "source_job_uid": "CryoSPARC source job, chosen by the model",
                "source_output": "CryoSPARC output group, chosen by the model",
            }
        },
        "reason": "Decision reason. Use your own evidence only.",
        "confidence": "Number from 0.0 to 1.0.",
        "risk_flags": [],
        "evidence": [],
    },
    "valid_examples": [
        {
            "schema_version": "2.0",
            "decision_type": "stop",
            "reason": "No safe autonomous action is clear.",
            "confidence": 0.5,
            "risk_flags": ["needs_human_review"],
            "evidence": [],
        },
        {
            "schema_version": "2.0",
            "decision_type": "forward",
            "action": "select_2D",
            "job_type": "select_2D",
            "parameters": {"selected_templates": "0,3,7"},
            "connections": {
                "particles": {"source_job_uid": "J103", "source_output": "particles"},
                "templates": {"source_job_uid": "J103", "source_output": "class_averages"},
            },
            "reason": "Select the best completed Class 2D branch and continue with one ordinary Select 2D Job.",
            "confidence": 0.9,
            "risk_flags": [],
            "evidence": ["Visual comparison identifies the best completed branch."],
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--current-node")
    parser.add_argument("--dataset-json", default="{}")
    parser.add_argument("--dataset-json-file")
    parser.add_argument("--known-workflow-dir", action="append", default=[])
    parser.add_argument("--backend", choices=["local", "api"], default="local")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--api-model", default=os.getenv("OPENAI_MODEL", "openai/gpt-6.1-sol"))
    parser.add_argument(
        "--fixed-api-model",
        default=os.getenv("CRYOAGENT_FIXED_API_MODEL", "openai/gpt-6-luna"),
        help=(
            "Cheaper API model for deterministic execution steps. Planning, Inspect Picks, "
            "and Select 2D continue to use --api-model."
        ),
    )
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-prompt-cache-mode", choices=["explicit", "implicit", "disabled"], default="explicit")
    parser.add_argument("--api-prompt-cache-key")
    parser.add_argument("--api-prompt-cache-ttl", default="30m")
    parser.add_argument("--model-python", default=DEFAULT_MODEL_PYTHON)
    parser.add_argument("--model-srun-prefix")
    parser.add_argument("--server-python", default=DEFAULT_SERVER_PYTHON)
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--mcp-server", default=DEFAULT_MCP_SERVER)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument(
        "--planning-mode", choices=["off", "deterministic", "api"], default="off",
        help="Add a whole-workspace experiment planner before each execution decision.",
    )
    parser.add_argument("--planner-max-actions", type=int, default=4)
    parser.add_argument("--guidance-cards", default=str(DEFAULT_CARDS_PATH),
                        help="Reviewed, advisory-only supplemental experience JSON.")
    parser.add_argument("--disable-stage-guidance", action="store_true",
                        help="Keep the original prompts without supplemental experience.")
    parser.add_argument("--guidance-max-bytes", type=int, default=8192,
                        help="Maximum UTF-8 bytes of optional advice per API prompt.")
    parser.add_argument("--max-kb-tool-calls", type=int, default=4, help="Maximum read-only KB calls per decision.")
    parser.add_argument("--kb-tool-policy", choices=["required", "auto", "disabled"], default="required", help="KB policy: require, allow, or strictly disable KB tools.")
    parser.add_argument(
        "--kb-escalation-after-jobs",
        type=int,
        default=0,
        help=(
            "For a single ablation run, keep the configured KB policy until this many "
            "workspace jobs exist. If the observer then reports an unmet target resolution, "
            "run one bounded KB-guided optimization decision. Zero disables this behavior."
        ),
    )
    parser.add_argument(
        "--kb-escalation-max-rounds",
        type=int,
        default=1,
        help="Maximum execution decisions in the post-budget KB optimization phase.",
    )
    parser.add_argument("--max-validation-failures", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--wait-timeout-seconds", type=int, default=43200)
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--wait-retries", type=int, default=3, help="Retries for transient result-package failures.")
    parser.add_argument("--wait-retry-delay-seconds", type=int, default=15)
    parser.add_argument("--resume-checkpoint", help="Resume from a previously written checkpoint JSON.")
    parser.add_argument("--extract-review", action=argparse.BooleanOptionalAction, default=True,
                        help="Review bounded extraction pilots before full-data extraction (API backend).")
    parser.add_argument("--extract-review-micrographs", type=int, default=6)
    parser.add_argument("--user-control-dir", help="Optional durable operator commands, applied between completed jobs and decisions")
    parser.add_argument("--output-dir", default="reports/autonomous_mcp_closed_loop")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Attach relevant visual evidence; local backend uses the multimodal model worker.",
    )
    parser.add_argument(
        "--stop-after-job-type",
        help="Stop after this successfully completed job type is observed.",
    )
    parser.add_argument(
        "--vision-max-classes",
        type=int,
        default=0,
        help="Maximum Class 2D images attached for visual selection; 0 attaches all classes.",
    )
    parser.add_argument(
        "--cryosift",
        action="store_true",
        help="Run CryoSift on completed Class 2D jobs before Select 2D decisions.",
    )
    parser.add_argument("--cryosift-threshold", type=float, default=3.0)
    parser.add_argument("--cryosift-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--proshade-symmetry",
        action="store_true",
        help="Attach CPU ProSHADE symmetry evidence after a completed 3D reconstruction.",
    )
    parser.add_argument("--proshade-resolution-A", type=float, default=8.0)
    parser.add_argument("--proshade-find-symmetry-center", action=argparse.BooleanOptionalAction,
                        default=True, help="Find the symmetry center before detection (default: enabled).")
    parser.add_argument("--proshade-timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def resolve_kb_escalation(
    args: argparse.Namespace,
    observation: dict[str, Any],
    state: dict[str, Any],
) -> tuple[str, bool, dict[str, Any]]:
    """Activate one KB-assisted recovery decision only after the no-KB budget is exhausted."""
    normalized = {
        "activated": bool(state.get("activated", False)),
        "rounds_completed": int(state.get("rounds_completed", 0)),
        "trigger_job_count": state.get("trigger_job_count"),
        "trigger_best_resolution_A": state.get("trigger_best_resolution_A"),
    }
    if args.kb_escalation_after_jobs <= 0:
        return args.kb_tool_policy, False, normalized

    dag = observation.get("dag") or {}
    job_count = len(dag.get("nodes") or [])
    issues = observation.get("observed_issues") or []
    target_unmet = next(
        (issue for issue in issues if issue.get("issue_type") == "target_resolution_unmet"),
        None,
    )
    if (
        not normalized["activated"]
        and job_count >= args.kb_escalation_after_jobs
        and target_unmet is not None
    ):
        normalized["activated"] = True
        normalized["trigger_job_count"] = job_count
        normalized["trigger_best_resolution_A"] = (
            (target_unmet.get("evidence") or {}).get("best_observed_resolution_A")
        )

    active = (
        normalized["activated"]
        and normalized["rounds_completed"] < args.kb_escalation_max_rounds
    )
    return ("required" if active else args.kb_tool_policy), active, normalized


async def main_async() -> None:
    args = parse_args()
    args.stage_guidance = (
        SupplementalGuidance(args.guidance_cards, max_bytes=args.guidance_max_bytes)
        if args.backend == "api" and not args.disable_stage_guidance else None
    )
    run_dir = make_run_dir(Path(args.output_dir))
    dataset_info = load_dataset_info(args)
    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "project_uid": args.project,
        "workspace_uid": args.workspace,
        "start_current_node": args.current_node,
        "backend": args.backend,
        "model_paths": {
            "base_model": args.base_model if args.backend == "local" else None,
            "adapter": args.adapter if args.backend == "local" else None,
            "api_base": args.api_base if args.backend == "api" else None,
            "api_model": args.api_model if args.backend == "api" else None,
            "fixed_api_model": args.fixed_api_model if args.backend == "api" else None,
        },
        "command": " ".join(sys.argv),
        "rounds": [],
        "planning_mode": args.planning_mode,
        "stage_guidance": args.stage_guidance.metadata() if args.stage_guidance else {"enabled": False},
        "experiment_history": [],
        "symmetry_evidence_history": {},
        "extract_reviews": {},
        "completion": {},
        "kb_escalation": {
            "activated": False,
            "rounds_completed": 0,
            "trigger_job_count": None,
            "trigger_best_resolution_A": None,
        },
        "stop_reason": None,
        "success": False,
    }

    server_env = os.environ.copy()
    symmetry_memory_file = Path(args.output_dir).resolve() / f"{args.project}_{args.workspace}_symmetry_memory.json"
    server_env["CRYOAGENT_SYMMETRY_EVIDENCE_FILE"] = str(symmetry_memory_file)
    server_env.setdefault("CRYOAGENT_PROSHADE_CACHE_DIR", str(run_dir.resolve() / "proshade"))
    # The autonomous runner uses one deterministic path until the first
    # refinement; generic MCP callers retain the full candidate registry.
    server_env.setdefault("CRYOAGENT_FIXED_INITIAL_FUNNEL", "1")
    if os.getenv("CRYOAGENT_GPU_LANE"):
        server_env["CRYOAGENT_GPU_LANE"] = os.environ["CRYOAGENT_GPU_LANE"]
    server_params = StdioServerParameters(
        command=args.server_python,
        args=[str(Path(args.project_dir) / args.mcp_server)],
        cwd=args.project_dir,
        env=server_env,
    )

    resume_checkpoint = None
    if args.resume_checkpoint:
        resume_checkpoint = json.loads(Path(args.resume_checkpoint).read_text())
        summary["experiment_history"] = list(resume_checkpoint.get("experiment_history") or [])
        summary["kb_escalation"].update(resume_checkpoint.get("kb_escalation") or {})
        summary["extract_reviews"] = dict(resume_checkpoint.get("extract_reviews") or {})
        summary["completion"] = dict(resume_checkpoint.get("completion") or {})
        summary["previous_stage_guidance"] = resume_checkpoint.get("stage_guidance")
        for evidence in (resume_checkpoint.get("symmetry_evidence_history") or {}).values():
            remember_evidence(symmetry_memory_file, args.project, args.workspace, evidence)
    current_node = args.current_node
    if current_node is None and resume_checkpoint:
        pending_jobs = resume_checkpoint.get("pending_jobs") or []
        current_node = (pending_jobs[0].get("job_uid") if pending_jobs else None) or resume_checkpoint.get("current_node")
    summary["resume_checkpoint"] = args.resume_checkpoint
    summary["checkpoint_file"] = str(run_dir / "checkpoint.json")
    feedback_to_model: dict[str, Any] | None = None
    automatic_evidence = None
    automatic_evidence_node = None
    validation_failures = 0

    model_worker = None
    if args.backend == "local":
        model_worker = ModelWorker(args, run_dir / "model_worker.stderr.log")
        model_worker.start()
        summary["model_worker"] = {
            "command": model_worker.command,
            "load_event": model_worker.load_event,
        }
    else:
        summary["api_backend"] = {
            "api_base": args.api_base,
            "api_model": args.api_model,
            "fixed_api_model": args.fixed_api_model,
            "api_key_env": args.api_key_env,
        }
    async with AsyncExitStack() as stack:
        if model_worker is not None:
            stack.callback(model_worker.close)
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # A checkpoint written after submission names a pending Job as the
        # current node. Resume must wait for it instead of asking the model to
        # decide from a non-terminal state and immediately exiting.
        if resume_checkpoint and resume_checkpoint.get("pending_jobs"):
            resumed_results = []
            for job in resume_checkpoint["pending_jobs"]:
                job_uid = job.get("job_uid")
                if not job_uid:
                    continue
                resumed_results.append(await wait_for_job_result_with_retries(
                    session,
                    {
                        "project_uid": args.project,
                        "workspace_uid": args.workspace,
                        "job_uid": job_uid,
                        "timeout_seconds": args.wait_timeout_seconds,
                        "poll_interval_seconds": args.poll_interval_seconds,
                        "include_next_candidates": True,
                    },
                    retries=args.wait_retries,
                    retry_delay_seconds=args.wait_retry_delay_seconds,
                ))
            summary["resumed_pending_jobs"] = resumed_results
            terminal = [item for item in resumed_results if item.get("ready_for_model")]
            if len(terminal) != len(resumed_results):
                summary["stop_reason"] = "resume_pending_job_not_terminal"
                summary["finished_at"] = utc_now()
                write_json(run_dir / "summary.json", summary, None)
                print_run_summary(summary, run_dir)
                return
            if terminal:
                current_node = terminal[-1].get("job_uid") or current_node

        for round_index in range(1, args.max_rounds + 1):
            auxiliary_jobs = sum(len(r.get("jobs", [])) for r in summary["extract_reviews"].values())
            if round_index + auxiliary_jobs > args.max_rounds:
                summary["stop_reason"] = "max_rounds_reached"
                break
            from operator_control import decision_boundary

            async def validate_operator_target(job_uid):
                result = await call_tool_json(session, "get_job_result_package", {
                    "project_uid": args.project, "workspace_uid": args.workspace, "job_uid": job_uid,
                })
                if result.get("status") != "completed" or result.get("workspace_uid") != args.workspace:
                    raise ValueError("Operator target must be completed in the current workspace")

            intervention = await decision_boundary(getattr(args, "user_control_dir", None), current_node, validate_operator_target)
            if intervention["stop"]:
                summary["stop_reason"] = "user_stop"
                break
            current_node = intervention["current_node"]
            args.operator_guidance = intervention["guidance"]
            round_dir = run_dir / f"round_{round_index:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            round_log: dict[str, Any] = {
                "round": round_index,
                "started_at": utc_now(),
                "current_node": current_node,
                "files": {},
            }
            summary["rounds"].append(round_log)

            context_args = {
                "project_uid": args.project,
                "workspace_uid": args.workspace,
                "current_job_uid": current_node,
                "dataset_info": dataset_info,
                "known_workflow_dirs": args.known_workflow_dir or None,
            }
            model_input = await call_tool_json(
                session,
                "get_workflow_decision_context",
                context_args,
            )
            model_input["failure_context"] = build_failure_context(feedback_to_model)
            model_input["operator_guidance"] = getattr(args, "operator_guidance", None)
            history = read_memory(symmetry_memory_file, args.project, args.workspace)
            symmetry_evidence = next(reversed(history.values()), None)
            if (args.proshade_symmetry and dataset_info.get("symmetry_status") == "unknown"
                    and current_node and model_input.get("current_state", {}).get("last_action") in {
                        "homo_abinit", "homo_refine_new", "nonuniform_refine_new", "refine_3D_new"
                    } and model_input.get("current_state", {}).get("last_node_status") == "completed"):
                fresh_symmetry = await collect_symmetry_evidence(session, args, current_node)
                history = remember_evidence(symmetry_memory_file, args.project, args.workspace, fresh_symmetry)
                symmetry_evidence = fresh_symmetry if fresh_symmetry.get("success") else (symmetry_evidence or fresh_symmetry)
                write_json(round_dir / "symmetry_evidence.json", fresh_symmetry, round_log)
            summary["symmetry_evidence_history"] = history
            model_input["symmetry_evidence_history"] = history
            round_log["symmetry_evidence"] = symmetry_evidence
            candidate_context = await call_tool_json(
                session,
                "get_candidate_actions",
                {
                    "project_uid": args.project,
                    "workspace_uid": args.workspace,
                    "current_node_id": current_node,
                    "dataset_info": dataset_info,
                },
            )
            async def completion_job(uid):
                return await call_tool_json(session, "get_job_result_package", {
                    "project_uid": args.project, "workspace_uid": args.workspace, "job_uid": uid,
                    "include_next_candidates": False,
                })

            async def completion_observation():
                return await call_tool_json(session, "get_reconstruction_observation", {
                    "project_uid": args.project, "workspace_uid": args.workspace, "dataset_info": dataset_info,
                })

            current_info = (model_input.get("current_state") or {}).get("last_node_info") or {}
            completion = await resolve_completion(current_info, dataset_info.get("target_resolution_A"),
                candidate_context.get("candidate_actions", []), completion_job, completion_observation)
            summary["completion"] = completion
            round_log["completion"] = completion
            write_json(round_dir / "completion.json", completion, round_log)
            if completion["action"] == "wait_validation":
                current_node = completion["validation_job_uid"]
                waited = await wait_for_job_result_with_retries(session, {
                    "project_uid": args.project, "workspace_uid": args.workspace, "job_uid": current_node,
                    "timeout_seconds": args.wait_timeout_seconds, "poll_interval_seconds": args.poll_interval_seconds,
                    "include_next_candidates": False,
                }, retries=args.wait_retries, retry_delay_seconds=args.wait_retry_delay_seconds)
                if waited.get("ready_for_model"):
                    continue
                summary["stop_reason"] = "target_validation_needs_attention"
                break
            if completion["action"] in {"stop", "hold"}:
                summary["stop_reason"] = completion["stop_reason"]
                write_json(run_dir / "checkpoint.json", {
                    "status": "complete" if completion["action"] == "stop" else "needs_attention",
                    "project_uid": args.project, "workspace_uid": args.workspace, "current_node": current_node,
                    "pending_jobs": [], "completion": completion, "extract_reviews": summary["extract_reviews"],
                    "stage_guidance": summary["stage_guidance"],
                    "experiment_history": summary["experiment_history"], "kb_escalation": summary["kb_escalation"],
                    "symmetry_evidence_history": summary["symmetry_evidence_history"],
                }, None)
                break
            forced_completion_decision = None
            if os.getenv("CRYOAGENT_BENCHMARK_BUDGET_FILE") and completion["action"] in {"continue", "validate"}:
                from benchmark_budget import budget_status
                status = await asyncio.to_thread(budget_status)
                round_log["benchmark_budget"] = status
                if status["remaining"] == 0:
                    summary["stop_reason"] = "benchmark_job_budget_reached"
                    summary["benchmark_budget"] = status
                    break
            if completion["action"] == "validate":
                forced_completion_decision = validation_decision(completion)
                candidate_context = {**candidate_context, "candidate_actions": [completion["candidate"]]}
                model_input["candidate_actions"] = candidate_context["candidate_actions"]
            template_picker_scale_evidence = attach_template_picker_volume_scale_evidence(
                candidate_context, args.project,
            )
            if template_picker_scale_evidence:
                model_input["template_picker_volume_scale_evidence"] = template_picker_scale_evidence
            # Retain the complete candidate set for evidence collection. Planning
            # intentionally narrows executable actions, but must never hide Class
            # 2D/CryoSift evidence required to choose among rollback branches.
            evidence_candidate_context = candidate_context
            # A bounded plan is valid for one refinement-checkpoint decision only.
            # Never carry it into the deterministic Class 2D -> Select 2D ->
            # Ab-initio -> Refine funnel on the following loop iteration.
            planning_directive: dict[str, Any] = {}
            planning_checkpoint = (
                model_input.get("current_state", {}).get("last_action")
                in {"homo_refine_new", "nonuniform_refine_new", "refine_3D_new"}
            )
            if args.planning_mode != "off" and planning_checkpoint and not forced_completion_decision:
                observation, planning_directive, planner_trace = await build_round_plan(
                    session=session,
                    args=args,
                    dataset_info=dataset_info,
                    candidate_context=candidate_context,
                    experiment_history=summary["experiment_history"],
                    symmetry_evidence=symmetry_evidence,
                )
                candidate_context = restrict_candidate_context(candidate_context, planning_directive)
                model_input["candidate_actions"] = candidate_context["candidate_actions"]
                model_input["planning_directive"] = planning_directive
                round_log["planner_observation"] = observation
                round_log["planning_directive"] = planning_directive
                round_log["planner_trace"] = planner_trace
                write_json(round_dir / "planner_observation.json", observation, round_log)
                write_json(round_dir / "planning_directive.json", planning_directive, round_log)
                write_json(round_dir / "planner_trace.json", planner_trace, round_log)
            elif args.kb_escalation_after_jobs > 0:
                observation = await call_tool_json(session, "get_reconstruction_observation", {
                    "project_uid": args.project,
                    "workspace_uid": args.workspace,
                    "dataset_info": dataset_info,
                })
            else:
                observation = {}

            effective_kb_tool_policy, kb_escalation_active, kb_escalation = resolve_kb_escalation(
                args, observation, summary["kb_escalation"],
            )
            summary["kb_escalation"] = kb_escalation
            if (
                kb_escalation["activated"]
                and not kb_escalation_active
                and kb_escalation["rounds_completed"] >= args.kb_escalation_max_rounds
            ):
                summary["stop_reason"] = "kb_escalation_complete"
                round_log["next_round_reason"] = (
                    "The bounded post-20-job KB optimization decision has already completed."
                )
                break
            model_input["kb_escalation"] = {
                "active": kb_escalation_active,
                "job_threshold": args.kb_escalation_after_jobs,
                "job_count": len((observation.get("dag") or {}).get("nodes") or []),
                "target_resolution_A": dataset_info.get("target_resolution_A"),
                "trigger_best_resolution_A": kb_escalation.get("trigger_best_resolution_A"),
                "instruction": (
                    "This is the one bounded post-budget optimization decision. First query the "
                    "knowledge base for similar datasets, official guidance, and upstream recovery "
                    "evidence; then prioritize an evidence-backed upstream improvement over an "
                    "unchanged refinement repeat."
                    if kb_escalation_active
                    else "Knowledge-base escalation is not active for this decision."
                ),
            }
            round_log["kb_escalation"] = model_input["kb_escalation"]
            round_log["effective_kb_tool_policy"] = effective_kb_tool_policy
            round_log["model_input"] = model_input
            round_log["candidate_context"] = candidate_context
            write_json(round_dir / "model_input.json", model_input, round_log)
            write_json(round_dir / "candidate_actions.json", candidate_context, round_log)

            if model_input.get("internal_only") or model_input.get("ready_for_model") is False:
                # A user may resume from a manually created, still-running Job.
                # Treat it exactly like a Job submitted by this runner: wait for
                # the terminal result, then continue from its outputs without
                # asking the model to decide from partial state.
                if current_node:
                    wait_args = {
                        "project_uid": args.project,
                        "workspace_uid": args.workspace,
                        "job_uid": current_node,
                        "timeout_seconds": args.wait_timeout_seconds,
                        "poll_interval_seconds": args.poll_interval_seconds,
                        "include_next_candidates": True,
                    }
                    waited = await wait_for_job_result_with_retries(
                        session,
                        wait_args,
                        retries=args.wait_retries,
                        retry_delay_seconds=args.wait_retry_delay_seconds,
                    )
                    round_log["resumed_current_job"] = {
                        "arguments": wait_args,
                        "result": waited,
                    }
                    write_json(round_dir / "resumed_current_job.json", round_log["resumed_current_job"], round_log)
                    if waited.get("ready_for_model"):
                        current_node = waited.get("job_uid") or current_node
                        round_log["next_round_reason"] = "Existing running Job completed; continue from its outputs."
                        continue
                summary["stop_reason"] = "current_job_not_terminal"
                round_log["next_round_reason"] = model_input.get("message")
                break

            visual_context = None
            last_action = model_input.get("current_state", {}).get("last_action")
            if symmetry_evidence is None and args.proshade_symmetry and current_node and last_action in {
                "homo_abinit", "homo_refine_new", "nonuniform_refine_new", "refine_3D_new"
            }:
                symmetry_evidence = await call_tool_json(
                    session,
                    "analyze_volume_symmetry",
                    {
                        "project_uid": args.project,
                        "job_uid": current_node,
                        "resolution_A": args.proshade_resolution_A,
                        "find_symmetry_center": args.proshade_find_symmetry_center,
                        "timeout_seconds": args.proshade_timeout_seconds,
                    },
                )
                round_log["symmetry_evidence"] = symmetry_evidence
                write_json(round_dir / "symmetry_evidence.json", symmetry_evidence, round_log)
            if args.vision and current_node:
                candidate_types = {
                    item.get("job_type")
                    for item in candidate_context.get("candidate_actions", [])
                }
                rollback_actions = [
                    item for item in evidence_candidate_context.get("candidate_actions", [])
                    if item.get("quality_rollback")
                ]
                if rollback_actions:
                    comparison_uids = rollback_class2d_job_uids(rollback_actions)
                    contexts = []
                    for job_uid in comparison_uids:
                        context = await call_tool_json(
                            session,
                            "get_class_average_visual_context",
                            {
                                "project_uid": args.project,
                                "job_uid": job_uid,
                                "max_classes": args.vision_max_classes,
                            },
                        )
                        if args.cryosift:
                            context["cryosift"] = await call_tool_json(
                                session,
                                "evaluate_2d_classes_with_cryosift",
                                {
                                    "project_uid": args.project,
                                    "job_uid": job_uid,
                                    "threshold": args.cryosift_threshold,
                                    "timeout_seconds": args.cryosift_timeout_seconds,
                                },
                            )
                        contexts.append(context)
                    visual_context = {
                        "kind": "class_average_comparison",
                        "contexts": contexts,
                        "decision_fallback": (
                            "If image evidence is unavailable, use the per-branch CryoSift "
                            "scores and class statistics as structured evidence instead."
                        ),
                    }
                elif last_action == "class_2D_new" and (
                    "select_2D" in candidate_types or any(
                        c.get('diameter_trial') for c in candidate_context.get('candidate_actions', []))
                ):
                    comparison_uids = []
                    for item in candidate_context.get("candidate_actions", []):
                        pilot_class=(item.get('diameter_trial') or {}).get('class_job_uid')
                        if pilot_class and pilot_class not in comparison_uids:
                            comparison_uids.append(pilot_class)
                        if item.get("job_type") != "select_2D":
                            continue
                        for job_uid in item.get("class2d_comparison_group") or []:
                            if job_uid and job_uid not in comparison_uids:
                                comparison_uids.append(job_uid)
                    if len(comparison_uids) > 1:
                        contexts = []
                        for job_uid in comparison_uids:
                            context = await call_tool_json(
                                session,
                                "get_class_average_visual_context",
                                {
                                    "project_uid": args.project,
                                    "job_uid": job_uid,
                                    "max_classes": args.vision_max_classes,
                                },
                            )
                            if args.cryosift:
                                context["cryosift"] = await call_tool_json(
                                    session,
                                    "evaluate_2d_classes_with_cryosift",
                                    {
                                        "project_uid": args.project,
                                        "job_uid": job_uid,
                                        "threshold": args.cryosift_threshold,
                                        "timeout_seconds": args.cryosift_timeout_seconds,
                                    },
                                )
                            contexts.append(context)
                        visual_context = {
                            "kind": "class_average_comparison",
                            "contexts": contexts,
                        }
                    else:
                        visual_context = await call_tool_json(
                            session,
                            "get_class_average_visual_context",
                            {
                                "project_uid": args.project,
                                "job_uid": current_node,
                                "max_classes": args.vision_max_classes,
                            },
                        )
                        if args.cryosift:
                            visual_context["cryosift"] = await call_tool_json(
                                session,
                                "evaluate_2d_classes_with_cryosift",
                                {
                                    "project_uid": args.project,
                                    "job_uid": current_node,
                                    "threshold": args.cryosift_threshold,
                                    "timeout_seconds": args.cryosift_timeout_seconds,
                                },
                            )
                elif last_action == "select_2D" and any(
                    c.get('pilot_template_recovery') for c in candidate_context.get('candidate_actions', [])
                ):
                    from upstream_evidence import template_picker_visual_context
                    visual_context = await template_picker_visual_context(
                        call_tool_json, session, args.project, candidate_context,
                        cryosift=args.cryosift, threshold=args.cryosift_threshold,
                        timeout_seconds=args.cryosift_timeout_seconds)
                elif "blob_picker_gpu" in candidate_types and last_action == "patch_ctf_estimation_multi":
                    files = model_input.get("dataset_info", {}).get("available_input_files") or {}
                    micrograph_path = files.get("micrograph_blob_paths")
                    micrograph_root = None
                    if isinstance(micrograph_path, str) and micrograph_path:
                        micrograph_root = str(Path(micrograph_path).expanduser().parent)
                    visual_context = await call_tool_json(
                        session,
                        "get_micrograph_preview_visual_context",
                        {
                            "project_uid": args.project,
                            "job_uid": current_node,
                            "max_micrographs": 6,
                            "micrograph_root": micrograph_root,
                        },
                    )
                    visual_context["decision_stage"] = "blob_picker_diameter"
                elif "inspect_picks_v2" in candidate_types and last_action != "inspect_picks_v2" and (
                    last_action in {"blob_picker_gpu", "auto_blob_picker_gpu", "template_picker_gpu", "micrograph_junk_detector", "junk_detector_v1"}
                    or bool(planning_directive.get("approved_action_ids"))
                ):
                    inspect_candidates = [
                        item for item in candidate_context.get("candidate_actions", [])
                        if item.get("job_type") == "inspect_picks_v2"
                    ]
                    inspect_inputs = (inspect_candidates[0].get("required_inputs") or {}) if inspect_candidates else {}
                    particle_sources = inspect_inputs.get("particles") or []
                    visual_job_uid = (
                        particle_sources[0].get("source_job_uid")
                        if particle_sources and isinstance(particle_sources[0], dict)
                        else current_node
                    )
                    files = model_input.get("dataset_info", {}).get("available_input_files") or {}
                    micrograph_path = files.get("micrograph_blob_paths")
                    micrograph_root = None
                    if isinstance(micrograph_path, str) and micrograph_path:
                        micrograph_root = str(Path(micrograph_path).expanduser().parent)
                    visual_context = await call_tool_json(
                        session,
                        "get_pick_inspection_visual_context",
                        {
                            "project_uid": args.project,
                            "job_uid": visual_job_uid,
                            "max_micrographs": 8,
                            "max_picks_per_micrograph": 600,
                            "micrograph_root": micrograph_root,
                        },
                    )
                    visual_context["decision_stage"] = "inspect_picks_threshold"
                elif (
                    "extract_micrographs_multi" in candidate_types
                    and last_action == "inspect_picks_v2"
                ):
                    visual_context = await call_tool_json(session, 'get_inspect_pick_review_context', {
                        'project_uid': args.project, 'job_uid': current_node, 'max_micrographs': 3})
                    visual_context.setdefault('kind', 'inspect_result_review')
                    last_info = model_input.get("current_state", {}).get("last_node_info") or {}
                    groups = (last_info.get("inputs") or {}).get("groups") or []
                    source_job_uid = next(
                        (
                            uid
                            for group in groups
                            if group.get("name") == "particles"
                            for uid in group.get("connected_job_uids", [])
                        ),
                        None,
                    )
                    if source_job_uid:
                        files = model_input.get("dataset_info", {}).get("available_input_files") or {}
                        micrograph_path = files.get("micrograph_blob_paths")
                        micrograph_root = None
                        if isinstance(micrograph_path, str) and micrograph_path:
                            micrograph_root = str(Path(micrograph_path).expanduser().parent)
                        extract_visual_context = await call_tool_json(
                            session,
                            "get_extract_box_visual_context",
                            {
                                "project_uid": args.project,
                                "job_uid": source_job_uid,
                                "max_examples": 3,
                                "micrograph_root": micrograph_root,
                                "particle_job_uid": current_node,
                            },
                        )
                        # Keep box-size evidence in this same decision. Inspect
                        # revision must be evaluated before any extraction starts.
                        visual_context['extract_box_evidence'] = {
                            k: v for k, v in extract_visual_context.items()
                            if k not in {'evidence_images', 'contact_sheet', 'score_scatter'}}
                        visual_context.setdefault('evidence_images', []).extend(
                            (extract_visual_context.get('evidence_images') or [])[:2])
                if visual_context:
                    # Preserve MCP failures verbatim; a missing image must not hide its root cause.
                    write_json(round_dir / "visual_context.json", visual_context, round_log)
                    round_log["visual_context"] = {
                        "kind": visual_context.get("kind"),
                        "source": visual_context.get("source"),
                        "image_count": visual_context.get("image_count"),
                        "evidence_image_count": len(visual_context.get("evidence_images") or []),
                        "evidence_image_roles": [
                            item.get("role") for item in visual_context.get("evidence_images") or []
                        ],
                        "class_ids": visual_context.get("class_ids"),
                        "local_path": (visual_context.get("contact_sheet") or {}).get("local_path"),
                    }
            guidance_audit: dict[str, Any] = {}
            from unattended import enabled as unattended_enabled, GUIDANCE as UNATTENDED_GUIDANCE
            if current_node != automatic_evidence_node:
                automatic_evidence = None
            if unattended_enabled() and current_node and automatic_evidence is None:
                from unattended import gather
                automatic_evidence = await gather(call_tool_json, session, args.project,
                    args.workspace, current_node, candidate_context)
                automatic_evidence_node = current_node
                if automatic_evidence:
                    write_json(round_dir / 'upstream_evidence.json', automatic_evidence, round_log)
            messages = build_autonomous_prompt(
                model_input=model_input,
                candidate_context=candidate_context,
                round_index=round_index,
                visual_context=visual_context,
                symmetry_evidence=symmetry_evidence,
                mark_static_cache_breakpoint=args.backend == "api" and args.api_prompt_cache_mode == "explicit",
                kb_tool_policy=effective_kb_tool_policy,
                visual_sft_mode=args.backend == "local" and args.vision,
                stage_guidance=args.stage_guidance,
                guidance_audit=guidance_audit,
            )
            if unattended_enabled():
                messages.insert(0, {'role': 'system', 'content': UNATTENDED_GUIDANCE})
                if automatic_evidence:
                    from upstream_evidence import message_for
                    messages.append(message_for(automatic_evidence))
            if args.stage_guidance is not None:
                write_json(round_dir / "stage_guidance_executor.json", guidance_audit, round_log)
            messages_file = round_dir / "model_messages.json"
            if getattr(args, "operator_guidance", None):
                messages.append({"role": "user", "content": (
                    "Authenticated operator guidance for this workflow. Follow it where safe; "
                    "do not bypass input compatibility, numeric validation, compute budgets, or approval rules. "
                    + json.dumps(args.operator_guidance, ensure_ascii=False)
                )})
            from prompt_context import focus_class_attachments
            messages = focus_class_attachments(messages, visual_context)
            if args.vision and os.getenv('CRYOAGENT_VALIDATED_PILOT_REFERENCE'):
                from pilot_promotion import append_reference
                messages = append_reference(messages,candidate_context.get('candidate_actions',[]),
                    os.environ['CRYOAGENT_VALIDATED_PILOT_REFERENCE'],args.project,args.workspace)
            write_json(messages_file, messages, round_log)

            generation_file = round_dir / "model_generation.json"
            try:
                for record in summary["extract_reviews"].values():
                    if (record.get("current_node") == current_node and not record.get("consumed")
                            and record.get("operator_guidance") != getattr(args, "operator_guidance", None)):
                        record.update(consumed=True, superseded_by_operator=True)
                saved_review = next((r for r in summary["extract_reviews"].values()
                    if r.get("current_node") == current_node and not r.get("consumed")), None)
                if forced_completion_decision or saved_review:
                    cached_decision = forced_completion_decision or saved_review["decision"]
                    model_call = {"raw_text": json.dumps(cached_decision),
                                  "controller_decision": "target_validation" if forced_completion_decision else "resume_extract_review"}
                elif args.backend == "api":
                    api_key = resolve_api_key(args.api_key, args.api_key_env)
                    execution_model = select_execution_api_model(args, candidate_context)
                    round_log["execution_api_model"] = execution_model

                    api_call_count = 0

                    async def api_model_call(working_messages, tools):
                        nonlocal api_call_count
                        # Some OpenAI-compatible gateways reject tool_choice when
                        # the request does not contain a tools array (the disabled
                        # KB ablation intentionally sends no tools).
                        tool_choice = None
                        if tools:
                            tool_choice = (
                                "required"
                                if effective_kb_tool_policy == "required" and api_call_count == 0
                                else "auto"
                            )
                        api_call_count += 1
                        return await asyncio.to_thread(
                            run_openai_compatible_model,
                            working_messages,
                            args.api_base,
                            api_key,
                            execution_model,
                            args.max_new_tokens,
                            args.temperature,
                            300,
                            prompt_cache_key=resolve_prompt_cache_key(args),
                            prompt_cache_options=build_prompt_cache_options(args),
                            tools=tools,
                            tool_choice=tool_choice,
                        )

                    async def kb_executor(tool_name, arguments):
                        return await call_tool_json(session, tool_name, arguments)

                    stage_contract_active = (visual_context or {}).get("decision_stage") in {
                        "blob_picker_diameter", "inspect_picks_threshold", "extract_box_size"
                    }
                    if effective_kb_tool_policy == "disabled" or stage_contract_active:
                        # The ablation must not expose KB tools or perform KB calls.
                        model_call = await api_model_call(messages, [])
                        model_call["tool_trace"] = []
                        model_call["messages_after_tools"] = messages
                    else:
                        model_call = await run_kb_tool_call_loop(
                            messages,
                            api_model_call,
                            kb_executor,
                            max_tool_calls=args.max_kb_tool_calls,
                        )
                    model_call["request_id"] = f"round_{round_index:02d}"
                else:
                    assert model_worker is not None
                    model_call = model_worker.generate(
                        request_id=f"round_{round_index:02d}",
                        messages=messages,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
            except Exception as exc:
                error_payload = {
                    "round": round_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "api_base": args.api_base if args.backend == "api" else None,
                    "api_model": (
                        locals().get("execution_model", args.api_model)
                        if args.backend == "api" else None
                    ),
                }
                round_log["model_error"] = error_payload
                write_json(round_dir / "model_error.json", error_payload, round_log)
                summary["stop_reason"] = "model_call_error"
                summary["error"] = error_payload
                summary["finished_at"] = utc_now()
                write_json(run_dir / "summary.json", summary, None)
                print_run_summary(summary, run_dir)
                return
            round_log["model_call"] = model_call
            round_log["kb_tool_calls"] = model_call.get("tool_trace", [])
            write_json(round_dir / "kb_tool_calls.json", model_call.get("tool_trace", []), round_log)
            round_log["raw_model_output_file"] = str(generation_file)
            write_json(generation_file, model_call, round_log)
            generation = model_call
            raw_output = generation.get("raw_text", "")
            round_log["raw_model_output"] = raw_output

            try:
                decision = parse_model_decision_text(raw_output)
                if not (unattended_enabled() and decision.get('decision_type') in {'request_input', 'stop', 'complete'}) and (visual_context or {}).get("decision_stage") in {
                    "blob_picker_diameter", "inspect_picks_threshold", "extract_box_size"
                }:
                    decision = normalize_visual_sft_decision(
                        decision,
                        candidate_context,
                        current_node,
                        model_input,
                        visual_context,
                    )
                parse_result = {"success": True, "decision": decision}
            except Exception as exc:
                decision = None
                parse_result = {
                    "success": False,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "raw_model_output": raw_output,
                }
            round_log["json_parse"] = parse_result
            write_json(round_dir / "json_parse.json", parse_result, round_log)

            if decision is None:
                validation_failures += 1
                feedback_to_model = make_feedback(
                    round_index,
                    "json_parse_failed",
                    parse_result,
                    model_input,
                    candidate_context,
                )
                if validation_failures >= args.max_validation_failures:
                    summary["stop_reason"] = "validation_failure_limit"
                    break
                continue

            from v2_decision_adapter import normalize_requested_actions, match_candidate
            if decision.get('decision_type') in {'forward', 'branch'}:
                actions = normalize_requested_actions(decision)
                if actions:
                    for action in actions:
                        matched = match_candidate(action, candidate_context.get('candidate_actions') or [])
                        if matched is not None:
                            action.update(action_id=matched['action_id'], job_type=matched['job_type'])
                    decision = {**decision, 'selected_actions': actions}
            from diameter_trial import guard_pilot_continuation
            decision, pilot_audit = guard_pilot_continuation(decision,candidate_context,visual_context)
            if pilot_audit:
                write_json(round_dir / 'pilot_continuation_guard.json',pilot_audit,round_log)
            decision, selection_audit = guard_select_decision(decision, candidate_context, visual_context)
            if selection_audit:
                write_json(round_dir / "selection_quality_guard.json", {
                    "repairs": selection_audit, "executed_decision": decision,
                    "note": "Model original output remains in model_generation.json and raw_model_output.",
                }, round_log)

            if unattended_enabled() and decision.get('decision_type') == 'request_input':
                from unattended import consume_request, gather
                write_json(round_dir / 'model_decision.json', decision, round_log)
                if not consume_request(args.output_dir, args.project, args.workspace, current_node, decision):
                    summary['stop_reason'] = 'unattended_recovery_exhausted'
                    summary['error'] = {'error': 'Automatic evidence requests exhausted; no safe legal progress was selected.'}
                    round_log['next_round_reason'] = summary['error']['error']
                    break
                automatic_evidence = await gather(call_tool_json, session, args.project,
                    args.workspace, current_node, candidate_context, requested=decision.get('requested_inputs') or decision.get('reason'))
                automatic_evidence_node = current_node
                write_json(round_dir / 'automatic_requested_evidence.json', automatic_evidence, round_log)
                feedback_to_model = make_feedback(round_index, 'automatic_evidence_retrieved',
                    {'success': True, 'instruction': UNATTENDED_GUIDANCE}, model_input, candidate_context)
                validation_failures = 0
                continue

            if planning_directive:
                allowed, message = decision_respects_plan(decision, planning_directive)
                if not allowed:
                    validation_failures += 1
                    planner_validation = {"success": False, "message": message, "plan": planning_directive}
                    round_log["planner_decision_validation"] = planner_validation
                    write_json(round_dir / "planner_decision_validation.json", planner_validation, round_log)
                    feedback_to_model = make_feedback(
                        round_index, "planner_decision_outside_plan", planner_validation,
                        model_input, candidate_context, failed_decision=decision,
                    )
                    if validation_failures >= args.max_validation_failures:
                        summary["stop_reason"] = "validation_failure_limit"
                        break
                    continue

            decision_file = round_dir / "model_decision.json"
            write_json(decision_file, decision, round_log)

            validation_args = {
                "decision": decision,
                "project_uid": args.project,
                "workspace_uid": args.workspace,
                "current_node_id": current_node,
                "dataset_info": dataset_info,
            }
            validation = await call_tool_json(
                session,
                "validate_v2_model_decision",
                validation_args,
            )
            round_log["mcp_validation_tool"] = {
                "tool_name": "validate_v2_model_decision",
                "arguments": validation_args,
                "result": validation,
            }
            write_json(round_dir / "validation.json", validation, round_log)

            if not validation.get("success"):
                validation_failures += 1
                feedback_to_model = make_feedback(
                    round_index,
                    "v2_validation_failed",
                    validation,
                    model_input,
                    candidate_context,
                    failed_decision=decision,
                )
                if validation_failures >= args.max_validation_failures:
                    summary["stop_reason"] = "validation_failure_limit"
                    break
                continue

            validation_failures = 0
            decision_type = decision.get("decision_type")
            if decision_type in {"stop", "complete", "request_input"}:
                summary["stop_reason"] = f"model_{decision_type}"
                round_log["next_round_reason"] = decision.get("reason")
                break

            execution_args = {
                "decision": decision,
                "project_uid": args.project,
                "workspace_uid": args.workspace,
                "current_node_id": current_node,
                "dataset_info": dataset_info,
                "dry_run": False,
            }
            if args.extract_review and args.backend == "api":
                candidate = next((c for c in candidate_context.get("candidate_actions", [])
                                  if c.get("action_id") == decision.get("action_id")), {})
                if is_full_extraction(decision, candidate):
                    write_json(round_dir / "initial_model_decision.json", decision, round_log)
                    try:
                        decision = await run_extraction_preflight(
                            args, session, decision, candidate, current_node, dataset_info,
                            summary, run_dir, round_dir, round_index, round_log,
                        )
                    except Exception as exc:
                        summary["stop_reason"] = "extract_review_needs_attention"
                        round_log["extract_review_error"] = {"type": type(exc).__name__, "error": str(exc)}
                        summary["error"] = round_log["extract_review_error"]
                        write_json(round_dir / "extract_review_error.json", round_log["extract_review_error"], round_log)
                        break
                    execution_args["decision"] = decision
                    write_json(round_dir / "execution_decision.json", decision, round_log)
                    write_json(round_dir / "model_decision.json", decision, round_log)
                    boundary = await decision_boundary(getattr(args, "user_control_dir", None), current_node, validate_operator_target)
                    if boundary["stop"]:
                        summary["stop_reason"] = "user_stop"
                        break
                    if boundary["current_node"] != current_node or boundary["guidance"] != args.operator_guidance:
                        for record in summary["extract_reviews"].values():
                            if record.get("current_node") == current_node and not record.get("consumed"):
                                record.update(consumed=True, superseded_by_operator=True)
                        current_node = boundary["current_node"]
                        feedback_to_model = None
                        continue
            execution = await call_tool_json(
                session,
                "execute_v2_model_decision",
                execution_args,
            )
            automatic_evidence = None
            round_log["mcp_execution_tool"] = {
                "tool_name": "execute_v2_model_decision",
                "arguments": execution_args,
                "result": execution,
            }
            write_json(round_dir / "execution.json", execution, round_log)

            created_jobs = find_created_jobs(execution)
            for record in summary["extract_reviews"].values():
                if record.get("current_node") == current_node and record.get("status") == "ready" and created_jobs:
                    record.update(consumed=True, full_jobs=created_jobs)
                    write_json(Path(record["artifacts_dir"]) / "state.json", record, None)
            round_log["created_jobs"] = created_jobs
            if kb_escalation_active and created_jobs:
                # Count an execution decision, not its fan-out: a box-size experiment can
                # legitimately create several CryoSPARC jobs as one bounded intervention.
                summary["kb_escalation"]["rounds_completed"] += 1
                round_log["kb_escalation"]["rounds_completed"] = summary["kb_escalation"]["rounds_completed"]
            if args.planning_mode != "off" and created_jobs:
                entries = build_experiment_history_entries(
                    round_index, decision, candidate_context, created_jobs, planning_directive,
                )
                summary["experiment_history"].extend(entries)
                round_log["experiment_history_entries"] = entries
                write_json(round_dir / "experiment_history_entries.json", entries, round_log)
            write_json(
                run_dir / "checkpoint.json",
                {
                    "schema_version": "1.0",
                    "status": "waiting_for_jobs" if created_jobs else "execution_failed",
                    "updated_at": utc_now(),
                    "project_uid": args.project,
                    "workspace_uid": args.workspace,
                    "round": round_index,
                    "current_node": current_node,
                    "pending_jobs": created_jobs,
                    "experiment_history": summary["experiment_history"],
                    "kb_escalation": summary["kb_escalation"],
                    "symmetry_evidence_history": summary["symmetry_evidence_history"],
                    "extract_reviews": summary["extract_reviews"], "completion": summary["completion"],
                    "stage_guidance": summary["stage_guidance"],
                    "next_round": round_index + 1,
                },
                None,
            )
            if not created_jobs:
                feedback_to_model = make_feedback(
                    round_index,
                    "execution_failed_or_no_job",
                    execution,
                    model_input,
                    candidate_context,
                    failed_decision=decision,
                )
                round_log["next_round_reason"] = "Execution returned no created job; feedback sent to model."
                continue

            job_feedbacks = []
            for job in created_jobs:
                wait_args = {
                    "project_uid": args.project,
                    "workspace_uid": args.workspace,
                    "job_uid": job["job_uid"],
                    "timeout_seconds": args.wait_timeout_seconds,
                    "poll_interval_seconds": args.poll_interval_seconds,
                    "include_next_candidates": True,
                }
                job_result = await wait_for_job_result_with_retries(
                    session,
                    wait_args,
                    retries=args.wait_retries,
                    retry_delay_seconds=args.wait_retry_delay_seconds,
                )
                job_result["local_job_logs"] = collect_job_logs(job["job_uid"])
                job_feedbacks.append(
                    {
                        "mcp_tool_name": "wait_for_job_result_package",
                        "arguments": wait_args,
                        "result": job_result,
                    }
                )
            round_log["job_results"] = job_feedbacks
            write_json(round_dir / "job_results.json", job_feedbacks, round_log)

            terminal_jobs = [
                feedback["result"]
                for feedback in job_feedbacks
                if feedback["result"].get("ready_for_model")
            ]
            if not terminal_jobs:
                summary["stop_reason"] = "job_wait_timeout_or_not_terminal"
                feedback_to_model = make_feedback(
                    round_index,
                    "job_not_terminal",
                    job_feedbacks,
                    model_input,
                    candidate_context,
                    failed_decision=decision,
                )
                break

            last_job = terminal_jobs[-1]
            current_node = last_job.get("job_uid") or current_node
            post_completion = None
            if last_job.get("job_type") == "validation":
                post_completion = await resolve_completion(last_job, dataset_info.get("target_resolution_A"),
                    [], completion_job, completion_observation)
            elif round_index + sum(len(r.get("jobs", [])) for r in summary["extract_reviews"].values()) >= args.max_rounds:
                post_completion = completion_check(last_job, dataset_info.get("target_resolution_A"), [])
            if post_completion and post_completion["action"] == "stop":
                summary["completion"] = post_completion
            write_json(
                run_dir / "checkpoint.json",
                {
                    "schema_version": "1.0",
                    "status": "complete" if post_completion and post_completion["action"] == "stop" else "ready_for_next_decision",
                    "updated_at": utc_now(),
                    "project_uid": args.project,
                    "workspace_uid": args.workspace,
                    "round": round_index,
                    "current_node": current_node,
                    "pending_jobs": [],
                    "experiment_history": summary["experiment_history"],
                    "kb_escalation": summary["kb_escalation"],
                    "symmetry_evidence_history": summary["symmetry_evidence_history"],
                    "extract_reviews": summary["extract_reviews"], "completion": summary["completion"],
                    "stage_guidance": summary["stage_guidance"],
                    "next_round": round_index + 1,
                },
                None,
            )
            feedback_to_model = make_feedback(
                round_index,
                "job_result",
                last_job,
                model_input,
                candidate_context,
                failed_decision=decision if is_failed_job_result(last_job) else None,
            )
            round_log["next_round_reason"] = (
                f"此步骤由第 {round_index} 轮 Model 输出中的 action 字段触发。MCP 只负责执行。"
            )
            if post_completion and post_completion["action"] == "stop":
                summary["stop_reason"] = post_completion["stop_reason"]
                round_log["completion"] = post_completion
                write_json(round_dir / "completion.json", post_completion, round_log)
                break
            if (
                args.stop_after_job_type
                and any(job.get("job_type") == args.stop_after_job_type for job in created_jobs)
                and not is_failed_job_result(last_job)
            ):
                summary["stop_reason"] = f"stop_after_{args.stop_after_job_type}_completed"
                round_log["stop_after_job_type"] = args.stop_after_job_type
                break

        else:
            summary["stop_reason"] = "max_rounds_reached"

    summary["finished_at"] = utc_now()
    summary["success"] = (
        summary["stop_reason"] in {
            "model_stop", "model_complete", "model_request_input", "max_rounds_reached", "target_resolution_met"
        }
        or str(summary["stop_reason"] or "").startswith("stop_after_")
    )
    write_json(run_dir / "summary.json", summary, None)
    print_run_summary(summary, run_dir)


async def run_extraction_preflight(args, session, decision, candidate, current_node, dataset_info,
                                   summary, run_dir, round_dir, round_index, round_log):
    from extract_review_evidence import build_evidence, PILOT_TITLE_PREFIX
    from cryosparc_client import cryosparc_client

    if not 1 <= args.extract_review_micrographs <= 20:
        raise ReviewBlocked("Extraction review micrograph budget must be within 1..20")
    original = copy.deepcopy(decision)
    parameters = {**(candidate.get("default_parameters") or {}), **(original.get("parameters") or {})}
    original.setdefault("parameters", {})["box_size_pix"] = parameters["box_size_pix"]
    from extract_review import resolve_review_connections
    connections = resolve_review_connections(original, candidate)
    original["connections"] = connections
    source = connections.get("particles")
    if not isinstance(source, dict) or not source.get("source_job_uid"):
        raise ReviewBlocked("Extraction review requires an unambiguous explicit particle source")
    saved = next(((k, r) for k, r in summary["extract_reviews"].items()
                  if r.get("current_node") == current_node and not r.get("consumed")), None)
    key = saved[0] if saved else review_key(current_node, original) + f"_{len(summary['extract_reviews'])}"
    state = summary["extract_reviews"].setdefault(key, {"current_node": current_node})
    state.setdefault("operator_guidance", copy.deepcopy(getattr(args, "operator_guidance", None)))
    directory = Path(state.setdefault("artifacts_dir", str(run_dir / "extract_reviews" / key)))
    directory.mkdir(parents=True, exist_ok=True)

    def persist():
        write_json(directory / "state.json", state, None)
        write_json(run_dir / "checkpoint.json", {
            "schema_version": "1.0", "status": "extract_review_pending", "updated_at": utc_now(),
            "project_uid": args.project, "workspace_uid": args.workspace, "current_node": current_node,
            "pending_jobs": [], "round": round_index, "next_round": round_index,
            "extract_reviews": summary["extract_reviews"], "completion": summary["completion"],
            "stage_guidance": summary.get("stage_guidance", {"enabled": False}),
            "experiment_history": summary["experiment_history"], "kb_escalation": summary["kb_escalation"],
            "symmetry_evidence_history": summary.get("symmetry_evidence_history", {}),
        }, None)

    async def submit(pilot_decision, label):
        arguments = {"decision": pilot_decision, "project_uid": args.project,
                     "workspace_uid": args.workspace, "current_node_id": current_node, "dataset_info": dataset_info}
        checked = await call_tool_json(session, "validate_v2_model_decision", arguments)
        write_json(directory / f"{label}_validation.json", checked, None)
        if not checked.get("success"):
            raise ReviewBlocked("MCP rejected bounded extraction pilot parameters/connections")
        execution = await call_tool_json(session, "execute_v2_model_decision", {**arguments, "dry_run": False})
        write_json(directory / f"{label}_execution.json", execution, None)
        jobs = find_created_jobs(execution)
        if len(jobs) != 1:
            raise ReviewBlocked("Pilot submission did not return one unambiguous job")
        return jobs[0]["job_uid"]

    async def wait(uid):
        def mark():
            cryosparc_client().find_project(args.project).find_job(uid).set_title(f"{PILOT_TITLE_PREFIX} {key} {uid}")
        await asyncio.to_thread(mark)
        return await wait_for_job_result_with_retries(session, {
            "project_uid": args.project, "workspace_uid": args.workspace, "job_uid": uid,
            "timeout_seconds": args.wait_timeout_seconds, "poll_interval_seconds": args.poll_interval_seconds,
            "include_next_candidates": False,
        }, retries=args.wait_retries, retry_delay_seconds=args.wait_retry_delay_seconds)

    async def observe(current, alternative):
        phase = "paired" if alternative else "initial"
        output = directory / phase
        observation = await call_tool_json(session, "get_reconstruction_observation", {
            "project_uid": args.project, "workspace_uid": args.workspace, "dataset_info": dataset_info,
        })
        content, facts = await asyncio.to_thread(build_evidence, args.project, current,
            source["source_job_uid"], output, alternative, historical_extract_results(observation))
        if getattr(args, "operator_guidance", None):
            facts["operator_guidance"] = args.operator_guidance
            content[0]["text"] = json.dumps(facts)
        messages = [{"role": "system", "content": REVIEW_INSTRUCTIONS}, {"role": "user", "content": content}]
        result = await asyncio.to_thread(run_openai_compatible_model, messages, args.api_base,
            resolve_api_key(args.api_key, args.api_key_env), args.api_model,
            min(max(args.max_new_tokens, 2048), 8192), args.temperature, 300)
        write_json(output / "model_response.json", result, None)
        choices = (result.get("raw_response") or {}).get("choices") or []
        if choices and choices[0].get("finish_reason") == "length":
            raise ReviewBlocked("Extraction review response was truncated; no size change was applied")
        parsed = parse_model_decision_text(result["raw_text"])
        if parsed.get("action") == "use_alternative":
            valid_ids = {s["sample_id"] for s in facts["samples"]}
            if not set(parsed.get("supporting_sample_ids") or []).issubset(valid_ids):
                raise ReviewBlocked("Review cited sample IDs absent from the paired evidence")
        return parsed

    all_aux = sum(len(r.get("jobs", [])) for r in summary["extract_reviews"].values())
    remaining = args.max_rounds - round_index + 1 - (all_aux - len(state.get("jobs", [])))
    amended = await review_extraction(state, original, submit, wait, observe, persist,
                                     remaining, args.extract_review_micrographs)
    # Revalidate the corrected FULL decision; do not leak num_extract from pilots.
    checked = await call_tool_json(session, "validate_v2_model_decision", {
        "decision": amended, "project_uid": args.project, "workspace_uid": args.workspace,
        "current_node_id": current_node, "dataset_info": dataset_info,
    })
    write_json(directory / "full_decision_validation.json", checked, None)
    if not checked.get("success"):
        raise ReviewBlocked("MCP rejected the reviewed full-data extraction decision")
    round_log["extract_review"] = {"state_file": str(directory / "state.json"),
                                   "initial_box": original["parameters"]["box_size_pix"],
                                   "final_box": amended["parameters"]["box_size_pix"]}
    return amended


def load_dataset_info(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset_json_file:
        return json.loads(Path(args.dataset_json_file).read_text())
    return json.loads(args.dataset_json)


def normalize_visual_sft_decision(
    decision: dict[str, Any],
    candidate_context: dict[str, Any],
    current_node: str | None,
    model_input: dict[str, Any],
    visual_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bridge the visual SFT target schema to the live V2/MCP action contract."""
    normalized = dict(decision)
    task_action = normalized.get("action") or normalized.get("job_type")
    if not task_action:
        task_action = normalized.get("decision")
    if task_action not in {
        "blob_picker_gpu", "inspect_picks_v2", "extract_micrographs_multi"
    }:
        last_action = (model_input.get("current_state") or {}).get("last_action")
        task_action = {
            "patch_ctf_estimation_multi": "blob_picker_gpu",
            "blob_picker_gpu": "inspect_picks_v2",
            "inspect_picks_v2": "extract_micrographs_multi",
        }.get(last_action, task_action)

    parameters = dict(normalized.get("parameters") or {})
    if task_action == "blob_picker_gpu":
        aliases = {
            "diameter_min_A": "diameter",
            "diameter_max_A": "diameter_max",
        }
        for old, new in aliases.items():
            if old in parameters and new not in parameters:
                parameters[new] = parameters.pop(old)
        parameters = {
            key: parameters[key]
            for key in ("diameter", "diameter_max")
            if key in parameters
        }
        if "diameter" in parameters and "diameter_max" in parameters:
            if float(parameters["diameter"]) > float(parameters["diameter_max"]):
                raise ValueError("Blob Picker diameter minimum must not exceed diameter_max.")
    elif task_action == "inspect_picks_v2":
        parameters = {
            key: parameters[key]
            for key in ("ncc_score_min", "local_power_min", "local_power_max", "ncc_score_thresh", "lpower_thresh_min", "lpower_thresh_max")
            if key in parameters
        }
    elif task_action == "extract_micrographs_multi":
        parameters = {key: parameters[key] for key in ("box_size_pix",) if key in parameters}
    aliases = {
        "ncc_score_min": "ncc_score_thresh",
        "local_power_min": "lpower_thresh_min",
        "local_power_max": "lpower_thresh_max",
    }
    for old, new in aliases.items():
        if old in parameters and new not in parameters:
            parameters[new] = parameters.pop(old)

    if task_action == "blob_picker_gpu" and visual_context:
        micrograph_count = visual_context.get("micrograph_count")
        if isinstance(micrograph_count, int) and micrograph_count > 0:
            # Blob Picker's CryoSPARC default may process a single micrograph.
            parameters["num_process"] = micrograph_count

    if normalized.get("decision_type") not in {"forward", "branch"}:
        normalized["decision_type"] = "forward"
    if task_action:
        normalized["action"] = task_action
        normalized["job_type"] = task_action
    normalized["parameters"] = parameters
    normalized.setdefault("reason", "Decision produced by the visual SFT adapter.")
    normalized.setdefault("confidence", 0.5)
    normalized.setdefault("risk_flags", [])
    normalized.setdefault("evidence", [])

    if task_action:
        candidates = [
            item
            for item in candidate_context.get("candidate_actions", [])
            if item.get("available") and item.get("job_type") == task_action
        ]
        forward_candidates = [
            item for item in candidates if item.get("action_type") == "forward"
        ]
        prefix = f"forward_{current_node}_" if current_node else None
        current_candidates = [
            item for item in forward_candidates
            if prefix and str(item.get("action_id", "")).startswith(prefix)
        ]
        explicit = [item for item in candidates
                    if item.get('action_id') == normalized.get('action_id')]
        pilot = [item for item in candidates
                 if (item.get('diameter_trial') or {}).get('role') == 'pilot']
        if task_action == 'blob_picker_gpu' and pilot:
            # Missing confidence is conservative for legacy/local model outputs.
            uncertain = normalized.get('diameter_confidence') != 'confident'
            choices = pilot if uncertain else [item for item in (current_candidates or forward_candidates)
                                              if not item.get('diameter_trial')]
        else:
            choices = explicit or current_candidates or forward_candidates
        if len(choices) == 1:
            normalized["action_id"] = choices[0]["action_id"]
            selected = choices[0]
            if selected.get('diameter_trial'):
                for key in ('num_process', 'num_extract'):
                    if key in (selected.get('default_parameters') or {}):
                        parameters[key] = selected['default_parameters'][key]
        elif len(choices) > 1:
            raise ValueError(
                f"Visual SFT action {task_action} is ambiguous at {current_node}; "
                "refusing to select a stale branch candidate."
            )
    return normalized


def build_autonomous_prompt(
    model_input: dict[str, Any],
    candidate_context: dict[str, Any],
    round_index: int,
    visual_context: dict[str, Any] | None = None,
    symmetry_evidence: dict[str, Any] | None = None,
    mark_static_cache_breakpoint: bool = False,
    kb_tool_policy: str = "required",
    visual_sft_mode: bool = False,
    stage_guidance: SupplementalGuidance | None = None,
    guidance_audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Keep stable decision instructions separate from per-round workflow state."""
    from prompt_context import compact_decision_payload

    def supplement(messages):
        if stage_guidance is None or visual_sft_mode:
            return messages
        enriched = stage_guidance.inject(
            messages, last_completed_job=last_completed_from_model_input(model_input),
            candidate_actions=candidate_context.get("candidate_actions") or [],
            workflow_state=model_input.get("current_state") or {},
            quality_evidence={"quality_assessment": candidate_context.get("quality_assessment") or {},
                              "visual_context": visual_context or {}},
            audience="executor", audit=guidance_audit,
        )
        return enriched
    if (visual_context or {}).get("visual_evidence_version") == "full_field_v4" and not visual_context.get("decision_stage"):
        visual_context = {**visual_context, "decision_stage": {
            "micrograph_preview": "blob_picker_diameter",
            "pick_inspection": "inspect_picks_threshold",
            "extract_box_comparison": "extract_box_size",
        }.get(visual_context.get("kind"))}
    if (visual_context or {}).get("decision_stage") in {
        "blob_picker_diameter", "inspect_picks_threshold", "extract_box_size"
    }:
        return supplement(build_visual_sft_prompt(
            model_input, visual_context, candidate_context, include_reason=not visual_sft_mode
        ))
    dynamic_payload = compact_decision_payload(model_input, candidate_context, round_index, symmetry_evidence)
    static_instructions = dict(STATIC_DECISION_INSTRUCTIONS)
    static_instructions["instruction"] += (
        " In reason, provide several concrete sentences of auditable rationale: cite the "
        "supplied state/image evidence, justify important parameter choices and their "
        "expected quality effects, and acknowledge uncertainty or missing evidence. "
        "Do not invent observations or provide private internal reasoning."
    )
    planning_directive = candidate_context.get("planning_directive")
    if planning_directive:
        static_instructions["instruction"] += (
            " Planning mode is active: choose only an action_id listed in planning_directive."
            " Include that exact action_id in your JSON. Do not substitute another candidate"
            " with the same job type."
        )
    diameter_reviews = [c['diameter_trial'] for c in candidate_context.get('candidate_actions', [])
                        if c.get('diameter_trial')]
    if diameter_reviews:
        from diameter_trial import GUIDANCE
        static_instructions['instruction'] += ' ' + GUIDANCE
    if kb_tool_policy == "disabled":
        static_instructions["instruction"] = static_instructions["instruction"].replace(
            "Before deciding, you must call at least one read-only kb_* tool to retrieve relevant similar cases, official job documentation, next-step statistics, or failure evidence. ",
            "Do not call any kb_* tools in this ablation; decide only from the current state, candidate actions, and available visual evidence. ",
        )
    static_text = json.dumps(
        static_instructions, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if mark_static_cache_breakpoint:
        static_content: Any = [{
            "type": "text",
            "text": static_text,
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }]
    else:
        static_content = static_text
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": static_content},
        {
            "role": "user",
            "content": json.dumps(dynamic_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        },
    ]
    if visual_context:
        if visual_context.get("kind") == "class_average_comparison":
            contexts = visual_context.get("contexts") or []
            content: list[dict[str, Any]] = [{
                "type": "text",
                "text": json.dumps({
                        "visual_instruction": (
                            "Compare every Class 2D contact sheet below. Each sheet is labelled by job_uid "
                            "and every tile by class_id. Select the best completed branch based on clear, "
                            "consistent particle views, intact particle boundaries, noise level, and view diversity. "
                            "For the chosen branch, inspect every labelled class and return its explicit "
                            "comma-separated selected_templates class-ID list. Do not use a fixed retention "
                            "fraction, an index range, or an automatic resolution cutoff. "
                            "If an image is unavailable, use the supplied CryoSift scores and class statistics "
                            "as structured evidence rather than requesting human input solely for missing images."
                            " " + CRYOSIFT_SCORE_GUIDANCE
                        ),
                    "comparison_jobs": [
                        {
                            "job_uid": item.get("source", {}).get("job_uid"),
                            "class_statistics": item.get("class_statistics", []),
                            "cryosift": item.get("cryosift"),
                        }
                        for item in contexts
                    ],
                }, ensure_ascii=False),
            }]
            for item in contexts:
                sheet = item.get("contact_sheet") or {}
                if sheet.get("data_url"):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": sheet["data_url"]},
                    })
        else:
            sheet = visual_context.get("contact_sheet") or {}
            image_url = sheet.get("data_url") if isinstance(sheet, dict) else None
            image_available = isinstance(image_url, str) and bool(image_url)
            visual_metadata = {
                key: value for key, value in visual_context.items()
                if key not in {"contact_sheet", "score_scatter", "evidence_images"}
            }
            if visual_context.get('kind') == 'inspect_result_review':
                from inspect_review import INSTRUCTIONS as inspect_review_instructions
                image_available = image_available or any(
                    item.get('data_url') for item in visual_context.get('evidence_images', []))
            if visual_context.get("score_scatter"):
                visual_metadata["score_scatter"] = {
                    key: value for key, value in visual_context["score_scatter"].items()
                    if key != "data_url"
                }
            decision_stage = visual_context.get("decision_stage")
            content = [
                {
                    "type": "text",
                    "text": json.dumps({
                        "visual_instruction": (
                            inspect_review_instructions
                        ) if visual_context.get('kind') == 'inspect_result_review' else (
                            'Review the source Class 2D sheet and the explicitly listed selected_templates '
                            'as candidate Template Picker templates, not a request to reselect classes. '
                            'Only the listed IDs are selected; other tiles provide comparison. '
                            'Use whole-particle appearance and calibrated upstream micrographs to choose '
                            'Template Picker diameter, or choose an offered alternative full-data route. '
                            'The preceding Blob minimum is not a measured whole-particle diameter. '
                            'Poor templates can correlate with background and produce millions of false or '
                            'overlapping hits. Do not call a provisional pilot diameter validated. '
                            + CRYOSIFT_SCORE_GUIDANCE
                        ) if visual_context.get('kind') == 'pilot_template_picker' else (
                            'Review these pilot Class 2D averages for particle-diameter validity. '
                            'Use an exact offered action_id: accept_full with the unchanged tested '
                            'diameter, retry a small Blob Picker, or Select 2D with explicit '
                            'selected_templates solely to supply relatively better templates for full-data '
                            'Template Picker. Compare images and CryoSift scores (lower is better); '
                            'an empty recommendation does not prohibit visually justified selections '
                            'above threshold, but explain the disagreement. Do not select junk or '
                            'reconstruct pilot particles. Compare every offered completed pilot route, '
                            'not only the latest. Select a relatively better executable route even if '
                            'all pilots have low absolute quality, explaining that risk honestly. '
                            'At the retry limit you must promote a route; do not stop or request input '
                            'solely because quality is poor.'
                            ' ' + CRYOSIFT_SCORE_GUIDANCE
                        ) if diameter_reviews and visual_context.get('kind') == 'class_average' else (
                            "Review the sampled pre-picker micrographs. Choose a plausible Blob Picker "
                            "particle diameter in Angstroms from repeated particle-like signal, not isolated "
                            "noise. Return the stage action and parameters.diameter; do not add a long explanation."
                        ) if visual_context.get("kind") == "micrograph_preview" else (
                            "Choose the Extract box_size_pix using acquisition pixel size, particle-size "
                            "context, and the attached Inspect Picks evidence. Preserve native sampling; "
                            "do not use Fourier crop. Return the extract action and parameters.box_size_pix."
                        ) if decision_stage == "extract_box_size" else (
                            "Review the NCC-Power scatter and micrograph overlay for Inspect Picks. "
                            "Keep calibrate_ncc=false and calibrate_pow=false for these raw scores. "
                            "Brightness is count density, not quality. The dense core may itself be junk; "
                            "there is no preferred retention fraction. A finite upper Power threshold is mandatory; "
                            "choose it from credible particle evidence, not automatically a high percentile. Use explicit thresholds, "
                            "not auto clustering. Return ncc_score_min and local_power_min; include "
                            "local_power_max is mandatory and must exceed local_power_min."
                        ) if visual_context.get("kind") == "pick_inspection" else (
                            "Review this class-average contact sheet for Select 2D. Every tile is "
                            "labelled class_id=<integer>. Use the image, per-class resolution statistics, "
                            "and CryoSift scores to choose every class to retain. You must return an explicit "
                            "comma-separated selected_templates list containing only labels shown here; do not "
                            "use a fixed retention fraction, class-index range, or automatic resolution cutoff."
                            " " + CRYOSIFT_SCORE_GUIDANCE
                        ) if visual_context.get("kind") == "class_average" else (
                            "Review the micrograph overlay and the NCC-versus-Power scatter plot for "
                            "Inspect Picks. Red circles mark Blob Picker locations. Use the scatter "
                            "distribution in raw units with calibrate_ncc=false and calibrate_pow=false. "
                            "Brightness is count density, not quality, and the core may be junk. "
                            "No retention fraction is preferred. Choose explicit NCC and Power thresholds "
                            "from credible particle evidence rather than automatically preserving the bright core. "
                            "An explicit finite local_power_max is mandatory. Do not use auto "
                            "clustering; this workflow requires explicit thresholds."
                        ),
                        "visual_context": visual_metadata,
                        "visual_attachment_status": (
                            "available" if image_available else
                            "unavailable; continue from structured state and candidate evidence without requesting human input solely for the missing overlay"
                        ),
                    }, ensure_ascii=False),
                },
            ]
            if isinstance(image_url, str) and image_url:
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            score_scatter = visual_context.get("score_scatter") or {}
            scatter_url = score_scatter.get("data_url") if isinstance(score_scatter, dict) else None
            if isinstance(scatter_url, str) and scatter_url:
                content.append({"type": "image_url", "image_url": {"url": scatter_url}})
            if visual_context.get('kind') == 'inspect_result_review':
                for item in visual_context.get('evidence_images', []):
                    if item.get('data_url'):
                        content.append({'type': 'text', 'text': item.get('description', item.get('role', 'Review evidence'))})
                        content.append({'type': 'image_url', 'image_url': {'url': item['data_url']}})
        messages.append({"role": "user", "content": content})
    return supplement(messages)


def build_visual_sft_prompt(
    model_input: dict[str, Any],
    visual_context: dict[str, Any] | None,
    candidate_context: dict[str, Any] | None = None,
    *, include_reason: bool = False,
) -> list[dict[str, Any]]:
    """Build the backend-neutral message contract for visual parameter decisions."""
    state = model_input.get("current_state") or {}
    dataset = model_input.get("dataset_info") or {}
    last_info = state.get("last_node_info") or {}
    stage = (visual_context or {}).get("decision_stage")
    acquisition = {
        "psize_A": dataset.get("pixel_size_A"),
        "accel_kv": dataset.get("accelerating_voltage_kv"),
        "cs_mm": dataset.get("spherical_aberration_mm"),
        "total_dose_e_per_A2": dataset.get("total_exposure_dose_e_per_A2"),
    }
    task_context: dict[str, Any] = {
        "decision_stage": stage,
        "dataset_id": dataset.get("empiar_id"),
        "acquisition_facts": {key: value for key, value in acquisition.items() if value is not None},
        "target_resolution_A": dataset.get("target_resolution_A"),
        "current_node_id": state.get("last_node_id"),
        "current_node_status": state.get("last_node_status"),
        "current_job_type": last_info.get("job_type"),
        "current_job_parameters": last_info.get("parameters"),
        "current_job_metrics": last_info.get("metrics"),
        "current_job_outputs": last_info.get("outputs"),
        "visual_context": {
            key: value for key, value in (visual_context or {}).items()
            if key not in {"contact_sheet", "score_scatter", "panels", "evidence_images"}
        },
    }
    task_context["candidate_actions"] = [
        {
            key: item.get(key)
            for key in ("action_id", "job_type", "description", "parameter_template", "required_inputs", "diameter_trial")
            if item.get(key) is not None
        }
        for item in (candidate_context or {}).get("candidate_actions", [])
        if item.get("available") and (
            item.get("job_type") == STAGE_ACTIONS.get(stage) or item.get('diameter_trial'))
    ]
    image_data_urls: list[str] = []
    for image_key in ("contact_sheet", "score_scatter"):
        image_info = (visual_context or {}).get(image_key) or {}
        data_url = image_info.get("data_url") if isinstance(image_info, dict) else None
        if data_url:
            image_data_urls.append(data_url)
    for image_info in (visual_context or {}).get("evidence_images", []):
        data_url = image_info.get("data_url") if isinstance(image_info, dict) else None
        if data_url:
            image_data_urls.append(data_url)
    return build_stage_decision_messages(
        stage,
        task_context,
        image_data_urls=image_data_urls,
        include_reason=include_reason,
    )


def select_execution_api_model(
    args: argparse.Namespace, candidate_context: dict[str, Any],
) -> str:
    """Reserve the stronger model for visual/parameter-sensitive decisions."""
    complex_job_types = {"inspect_picks_v2", "select_2D"}
    candidates = candidate_context.get("candidate_actions") or []
    if candidate_context.get('inspect_review_required'):
        return args.api_model
    if any(item.get("job_type") in complex_job_types for item in candidates):
        return args.api_model
    return args.fixed_api_model


async def collect_symmetry_evidence(session, args, current_node):
    return await call_tool_json(session, "analyze_volume_symmetry", {
        "project_uid": args.project, "job_uid": current_node,
        "resolution_A": args.proshade_resolution_A,
        "find_symmetry_center": args.proshade_find_symmetry_center,
        "timeout_seconds": args.proshade_timeout_seconds,
    })


async def build_round_plan(
    session: ClientSession,
    args: argparse.Namespace,
    dataset_info: dict[str, Any],
    candidate_context: dict[str, Any],
    experiment_history: list[dict[str, Any]],
    symmetry_evidence: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Observe, optionally query an API planner, then apply non-bypassable guards."""
    # A refine checkpoint selects exactly one next experiment. Its fixed
    # downstream funnel executes without further planning until the next FSC.
    planner_action_budget = 1
    observation = await call_tool_json(session, "get_reconstruction_observation", {
        "project_uid": args.project,
        "workspace_uid": args.workspace,
        "dataset_info": dataset_info,
    })
    observation["operator_guidance"] = getattr(args, "operator_guidance", None)
    if symmetry_evidence is not None:
        observation["symmetry_evidence"] = symmetry_evidence
    candidates = candidate_context.get("candidate_actions") or []
    if args.cryosift:
        cached = await asyncio.gather(*(
            asyncio.to_thread(
                load_cached_cryosift_result, args.project, job_uid, args.cryosift_threshold,
            )
            for job_uid in rollback_class2d_job_uids(candidates)
        ))
        observation["cryosift_evidence"] = [item for item in cached if item is not None]
    trace: dict[str, Any] = {"mode": args.planning_mode, "fallback_used": False}
    def planning_messages():
        audit: dict[str, Any] = {}
        messages = build_planning_prompt(
            observation, candidates, experiment_history, planner_action_budget,
            stage_guidance=getattr(args, "stage_guidance", None), guidance_audit=audit,
        )
        from unattended import enabled, GUIDANCE
        if enabled():
            messages.insert(0, {'role': 'system', 'content': GUIDANCE})
        if getattr(args, "stage_guidance", None) is not None:
            trace.setdefault("stage_guidance_prompts", []).append(audit)
        return messages

    plan: dict[str, Any]
    if args.planning_mode == "api":
        try:
            api_key = resolve_api_key(args.api_key, args.api_key_env)
            messages = planning_messages()
            response = await asyncio.to_thread(
                run_openai_compatible_model,
                messages, args.api_base, api_key, args.api_model,
                min(args.max_new_tokens, 1600), args.temperature, 300,
            )
            raw_plan = parse_model_decision_text(response.get("raw_text", ""))
            trace["initial_response"] = response
            plan = validate_plan(raw_plan, candidates, planner_action_budget)
            if plan["decision_type"] == "request_observations":
                trace['observation_request'] = {
                    'tasks': plan['observation_tasks'],
                    'normalization': plan.get('request_normalization'),
                }
                evidence = await call_tool_json(session, "execute_reconstruction_observation_tasks", {
                    "project_uid": args.project,
                    "workspace_uid": args.workspace,
                    "dataset_info": dataset_info,
                    "tasks": plan["observation_tasks"],
                })
                trace['observation_result'] = evidence
                if evidence.get('mcp_error') or evidence.get('success') is False:
                    raise ValueError('planner observation request failed; inspect observation_result')
                observation = {**observation, "planner_requested_evidence": evidence.get("results") or []}
                messages = planning_messages()
                response = await asyncio.to_thread(
                    run_openai_compatible_model,
                    messages, args.api_base, api_key, args.api_model,
                    min(args.max_new_tokens, 1600), args.temperature, 300,
                )
                raw_plan = parse_model_decision_text(response.get("raw_text", ""))
                trace["observation_response"] = response
                plan = validate_plan(raw_plan, candidates, planner_action_budget)
            if plan["decision_type"] == "request_observations":
                raise ValueError("planner requested a second observation turn without proposing an experiment")
        except Exception as exc:
            trace["fallback_used"] = True
            trace["planner_error"] = {"error_type": type(exc).__name__, "message": str(exc)}
            plan = build_policy_plan(observation, candidates, experiment_history, planner_action_budget)
    else:
        plan = build_policy_plan(observation, candidates, experiment_history, planner_action_budget)
    plan = apply_anti_refinement_loop(
        plan, observation, candidates, experiment_history, planner_action_budget,
    )
    if plan.get("decision_type") == "plan_experiments" and not plan.get("approved_action_ids") and candidates:
        trace["fallback_used"] = True
        trace["empty_plan_fallback"] = True
        plan = build_policy_plan(observation, candidates, experiment_history, planner_action_budget)
    return observation, plan, trace


def build_experiment_history_entries(
    round_index: int,
    decision: dict[str, Any],
    candidate_context: dict[str, Any],
    created_jobs: list[dict[str, Any]],
    planning_directive: dict[str, Any],
) -> list[dict[str, Any]]:
    """Persist a compact action-to-outcome link for later observer/model training."""
    by_id = {
        str(item.get("action_id")): item
        for item in candidate_context.get("candidate_actions") or []
    }
    selected = decision.get("selected_actions") or decision.get("actions") or [decision]
    created_by_type = {item.get("job_type"): item for item in created_jobs}
    entries = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        candidate = by_id.get(str(item.get("action_id"))) or {}
        job_type = item.get("job_type") or item.get("action") or candidate.get("job_type")
        entries.append({
            "round": round_index,
            "plan_objective": planning_directive.get("objective"),
            "action_id": item.get("action_id"),
            "job_type": job_type,
            "input_signature": _input_signature(candidate),
            "parameters": item.get("parameters") or {},
            "created_job_uid": (created_by_type.get(job_type) or {}).get("job_uid"),
        })
    return entries


def resolve_prompt_cache_key(args: argparse.Namespace) -> str | None:
    if args.api_prompt_cache_mode != "explicit":
        return None
    if args.api_prompt_cache_key:
        return args.api_prompt_cache_key
    return f"cryoagent:{args.project}:{args.workspace}:workflow-v2"


def build_prompt_cache_options(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.api_prompt_cache_mode != "explicit":
        return None
    return {"mode": "explicit", "ttl": args.api_prompt_cache_ttl}


class ModelWorker:
    def __init__(self, args: argparse.Namespace, stderr_path: Path):
        self.args = args
        self.stderr_path = stderr_path
        self.stderr_file = None
        worker_script = (
            "model_generate_vision_once.py"
            if args.vision
            else "model_generate_once.py"
        )
        self.command = [
            args.model_python,
            str(Path(args.project_dir) / "scripts" / worker_script),
            "--worker",
            "--base-model",
            args.base_model,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--device-map",
            args.device_map,
            "--torch-dtype",
            args.torch_dtype,
        ]
        if args.vision:
            self.command.extend(["--max-image-pixels", "200704"])
        if args.adapter:
            self.command.extend(["--adapter", args.adapter])
        if args.model_srun_prefix:
            self.command = shlex.split(args.model_srun_prefix) + self.command
        self.process: subprocess.Popen[str] | None = None
        self.load_event: dict[str, Any] | None = None
        self.requests_served = 0

    def start(self) -> None:
        self.stderr_file = self.stderr_path.open("w")
        self.process = subprocess.Popen(
            self.command,
            cwd=self.args.project_dir,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            bufsize=1,
        )
        self.load_event = self._read_event()
        if self.load_event.get("event") != "model_loaded":
            raise RuntimeError(f"Model worker failed to load: {self.load_event}")

    def generate(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        assert self.process is not None
        assert self.process.stdin is not None
        request = {
            "request_id": request_id,
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        event = self._read_event()
        if event.get("event") != "generation":
            raise RuntimeError(f"Unexpected model worker event: {event}")
        self.requests_served += 1
        event["success"] = True
        event["model_loaded_once"] = True
        event["worker_request_number"] = self.requests_served
        return event

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        self.process = None
        try:
            if process.stdin:
                process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                process.stdin.flush()
        except BrokenPipeError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=30)
        if self.stderr_file:
            self.stderr_file.close()

    def _read_event(self) -> dict[str, Any]:
        assert self.process is not None
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if not line:
                return {
                    "event": "worker_exited",
                    "returncode": self.process.poll(),
                    "stderr_path": str(self.stderr_path),
                }
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue


async def wait_for_job_result_with_retries(
    session: ClientSession,
    arguments: dict[str, Any],
    retries: int,
    retry_delay_seconds: int,
) -> dict[str, Any]:
    """Retry transient MCP/result-package failures without duplicating a Job."""
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max(retries, 0) + 2):
        try:
            result = await call_tool_json(session, "wait_for_job_result_package", arguments)
            if not result.get("mcp_error"):
                result["wait_attempts"] = attempts + [{"attempt": attempt, "status": "ok"}]
                return result
            attempts.append({"attempt": attempt, "status": "mcp_error", "error": result})
        except Exception as exc:
            attempts.append({
                "attempt": attempt,
                "status": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        if attempt <= max(retries, 0):
            await asyncio.sleep(max(retry_delay_seconds, 1) * attempt)
    return {
        "success": False,
        "ready_for_model": False,
        "wait_error": "wait_for_job_result_package failed after retries",
        "wait_attempts": attempts,
        "job_uid": arguments.get("job_uid"),
    }


async def call_tool_json(
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    result = await session.call_tool(tool_name, arguments)
    if result.isError:
        return {
            "success": False,
            "mcp_error": True,
            "tool_name": tool_name,
            "content": [content.model_dump() for content in result.content],
        }
    if not result.content:
        return {"success": False, "tool_name": tool_name, "content": []}
    first = result.content[0]
    text = getattr(first, "text", None)
    if text is None:
        return {
            "success": False,
            "tool_name": tool_name,
            "content": [content.model_dump() for content in result.content],
        }
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"success": False, "tool_name": tool_name, "raw_text": text}


def make_feedback(
    round_index: int,
    feedback_type: str,
    payload: Any,
    model_input: dict[str, Any],
    candidate_context: dict[str, Any],
    failed_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "message_type": "mcp_feedback_to_model",
        "round": round_index,
        "feedback_type": feedback_type,
        "payload": payload,
        "failed_decision": failed_decision,
        "current_workflow_state": model_input,
        "candidate_actions": candidate_context,
        "instruction": "Use this feedback to decide the next v2 JSON action. MCP only reports the result.",
    }


def build_failure_context(feedback: dict[str, Any] | None) -> dict[str, Any] | None:
    if not feedback:
        return None
    feedback_type = feedback.get("feedback_type")
    payload = feedback.get("payload")
    failed_decision = feedback.get("failed_decision")
    if feedback_type == "job_result" and not is_failed_job_result(payload):
        return None
    return {
        "has_failure": True,
        "failure_round": feedback.get("round"),
        "failure_stage": map_failure_stage(feedback_type, payload),
        "failed_decision": failed_decision,
        "failed_action": extract_failed_action(failed_decision),
        "execution_error": extract_execution_error(feedback_type, payload),
        "missing_inputs": extract_named_items(payload, ("missing_inputs", "missing_required_inputs")),
        "invalid_connections": extract_named_items(payload, ("invalid_connections", "connection_errors")),
        "available_inputs": extract_named_items(
            feedback.get("current_workflow_state"),
            ("available_inputs", "outputs", "input_sources"),
        ),
        "candidate_actions": feedback.get("candidate_actions"),
        "attempt_history": [
            {
                "round": feedback.get("round"),
                "action": (failed_decision or {}).get("action") or (failed_decision or {}).get("job_type"),
                "failure_stage": map_failure_stage(feedback_type, payload),
                "message": extract_error_message(payload),
            }
        ],
        "retry_guidance": build_retry_guidance(failed_decision, payload),
    }


def map_failure_stage(feedback_type: str | None, payload: Any) -> str:
    if feedback_type == "json_parse_failed":
        return "json_parse"
    if feedback_type == "v2_validation_failed":
        return "validation"
    if feedback_type == "job_not_terminal":
        return "timeout"
    if feedback_type == "execution_failed_or_no_job":
        text = json.dumps(payload, ensure_ascii=False, default=str).lower()
        if "approval_required" in text:
            return "approval"
        if "connect" in text or "input" in text:
            return "connect_inputs"
        if "queue" in text or "missing required" in text:
            return "queue"
        return "no_created_job"
    if feedback_type == "job_result":
        return "run"
    return feedback_type or "unknown"


def extract_failed_action(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    if not decision:
        return None
    return {
        "decision_type": decision.get("decision_type"),
        "action": decision.get("action"),
        "job_type": decision.get("job_type") or decision.get("action"),
        "parameters": decision.get("parameters") or {},
        "connections": decision.get("connections") or {},
    }


def extract_execution_error(feedback_type: str | None, payload: Any) -> dict[str, Any]:
    return {
        "source": infer_error_source(feedback_type, payload),
        "error_type": extract_error_type(payload),
        "message": extract_error_message(payload),
        "raw": payload,
    }


def infer_error_source(feedback_type: str | None, payload: Any) -> str:
    if feedback_type == "json_parse_failed":
        return "json_parser"
    text = json.dumps(payload, ensure_ascii=False, default=str).lower()
    if "cryosparc" in text or "http" in text or "traceback" in text:
        return "cryosparc"
    if "mcp" in text:
        return "mcp"
    return "runner"


def extract_error_type(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("error_type", "type", "code", "status_code"):
            if payload.get(key):
                return str(payload[key])
    return None


def extract_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("message", "error", "detail", "stderr", "raw_text"):
            value = payload.get(key)
            if value:
                return str(value)
        for value in payload.values():
            message = extract_error_message(value)
            if message:
                return message
    if isinstance(payload, list):
        for value in payload:
            message = extract_error_message(value)
            if message:
                return message
    return ""


def extract_named_items(payload: Any, names: tuple[str, ...]) -> list[Any]:
    found: list[Any] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in names:
                found.extend(value if isinstance(value, list) else [value])
            else:
                found.extend(extract_named_items(value, names))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(extract_named_items(value, names))
    return found


def build_retry_guidance(decision: dict[str, Any] | None, payload: Any) -> dict[str, Any] | None:
    action = (decision or {}).get("action") or (decision or {}).get("job_type")
    message = extract_error_message(payload)
    if not action or not message:
        return None
    lowered = message.lower()
    if "missing required" in lowered or "required input" in lowered:
        return {
            "retry_same_action_allowed": False,
            "blocked_actions": [action],
            "reason": message,
        }
    return {
        "retry_same_action_allowed": True,
        "blocked_actions": [],
        "reason": message,
    }


def is_failed_job_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    status = str(result.get("status") or result.get("job_status") or "").lower()
    if status in {"failed", "killed", "error"}:
        return True
    return bool(result.get("has_error") or result.get("error"))


def find_created_jobs(execution: dict[str, Any]) -> list[dict[str, Any]]:
    execution_result = execution.get("execution_result") or {}
    results = execution_result.get("execution_results") or []
    created = []
    for item in results:
        job_uid = item.get("job_uid")
        if job_uid and item.get('success') is True and (
                item.get('queued') is True or item.get('status') in
                {'queued', 'launched', 'started', 'running', 'waiting', 'completed'}):
            created.append(
                {
                    "job_uid": job_uid,
                    "job_type": item.get("job_type"),
                    "status": item.get("status"),
                    "queued": item.get("queued"),
                    "planned_action": item.get("planned_action"),
                }
            )
    return created


def collect_job_logs(job_uid: str) -> dict[str, Any]:
    roots = [
        Path("/home/share/cryoSPARC/P2") / job_uid,
        Path("/ssd1/linweifan/cryosparc/cryosparc_master/run") / job_uid,
    ]
    logs = {}
    for root in roots:
        if not root.exists():
            continue
        for name in ("job.log", "job.json", "events.bson"):
            path = root / name
            if path.exists() and path.is_file():
                try:
                    logs[str(path)] = path.read_text(errors="replace")[-200000:]
                except UnicodeDecodeError:
                    logs[str(path)] = "<binary log file>"
                except OSError as exc:
                    logs[str(path)] = f"<read failed: {exc}>"
    return logs


def make_run_dir(output_dir: Path) -> Path:
    run_dir = output_dir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Any, round_log: dict[str, Any] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    temporary.replace(path)
    if round_log is not None:
        round_log.setdefault("files", {})[path.name] = str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def print_run_summary(summary: dict[str, Any], run_dir: Path) -> None:
    print(f"Run directory: {run_dir}")
    for round_log in summary["rounds"]:
        print(f"[轮次 {round_log['round']}]")
        print(f"Model 输入：{round_log.get('files', {}).get('model_messages.json')}")
        print(f"Model 返回：{round_log.get('raw_model_output', '')[:500]}")
        validation = round_log.get("mcp_validation_tool", {}).get("result")
        print(f"JSON 校验：{json.dumps(validation, ensure_ascii=False, default=str)[:500]}")
        execution_tool = round_log.get("mcp_execution_tool", {})
        print(f"MCP 调用：{execution_tool.get('tool_name')}")
        print(f"cryoSPARC 结果：{json.dumps(round_log.get('job_results') or round_log.get('created_jobs') or round_log.get('mcp_execution_tool', {}).get('result'), ensure_ascii=False, default=str)[:500]}")
        print(f"下一轮原因：{round_log.get('next_round_reason')}")
        print(f"日志位置：{round_log.get('files')}")
    print(f"Stop reason: {summary['stop_reason']}")
    print(f"Summary: {run_dir / 'summary.json'}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
