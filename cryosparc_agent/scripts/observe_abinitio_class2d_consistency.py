"""Run the visual Ab-initio/Class 2D consistency observer for completed jobs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from abinitio_class2d_consistency import (
    build_abinitio_class2d_consistency_context,
    build_consistency_observer_messages,
)
from model_direct_runner import resolve_api_key, run_openai_compatible_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--jobs", required=True, help="Comma-separated completed homo_abinit job UIDs.")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-model", default="openai/gpt-5.6-sol")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output-dir", default="logs/abinitio_class2d_consistency")
    parser.add_argument("--max-selected-classes", type=int, default=30)
    return parser.parse_args()


def compact_context(context: dict) -> dict:
    """Keep provenance and local image path in logs without duplicating base64 PNG."""
    compact = dict(context)
    compact["contact_sheet"] = dict(context["contact_sheet"])
    compact["contact_sheet"].pop("data_url", None)
    return compact


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = resolve_api_key(args.api_key, args.api_key_env)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "project_uid": args.project, "results": []}
    for job_uid in (item.strip() for item in args.jobs.split(",")):
        if not job_uid:
            continue
        context = build_abinitio_class2d_consistency_context(
            args.project, job_uid, max_selected_classes=args.max_selected_classes,
        )
        messages = build_consistency_observer_messages(context)
        result = run_openai_compatible_model(
            messages, args.api_base, api_key, args.api_model,
            max_new_tokens=700, temperature=0.0, timeout_seconds=300,
        )
        try:
            verdict = json.loads(result.get("raw_text", ""))
        except json.JSONDecodeError as exc:
            verdict = {"status": "uncertain", "parse_error": str(exc), "raw_text": result.get("raw_text", "")}
        record = {
            "context": compact_context(context),
            "verdict": verdict,
            "model_generation": result,
        }
        record_path = output_dir / f"{args.project}_{job_uid}.json"
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["results"].append({"job_uid": job_uid, "record": str(record_path), "verdict": verdict})
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
