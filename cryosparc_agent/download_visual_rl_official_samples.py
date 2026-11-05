"""Download the fixed EMPIAR micrograph samples for a visual-RL batch.

Only the 20 selected single-frame MRC files per dataset are downloaded. The
download is resumable and every completed file is checked for its indexed size
and a plausible 2D MRC header before it is committed into the sample directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import struct
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "data/visual_rl_70_official_download_plan_20261001.json"
DEFAULT_OUTPUT = Path("/hdd1/linweifan/cryosparc/cryoagent_visual_rl/official_samples/visual-rl-100-datasets-20261001")
USER_AGENT = "CryoAgent-visual-RL-sample-downloader/1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def valid_2d_mrc(path: Path) -> tuple[bool, list[int] | None]:
    try:
        with path.open("rb") as stream:
            header = stream.read(1024)
        nx, ny, nz, mode = struct.unpack("<4i", header[:16])
        nsymbt = struct.unpack("<i", header[92:96])[0]
        bytes_per_voxel = {0: 1, 1: 2, 2: 4, 3: 4, 4: 8, 6: 2, 12: 2}.get(mode)
        dims = [nx, ny, nz]
    except (OSError, struct.error):
        return False, None
    valid = (
        0 < nx < 100000 and 0 < ny < 100000 and nz == 1
        and bytes_per_voxel is not None and nsymbt >= 0
        and path.stat().st_size >= 1024 + nsymbt + nx * ny * nz * bytes_per_voxel
    )
    return valid, dims


def download_one(item: dict[str, Any], destination: Path, retries: int) -> dict[str, Any]:
    expected_size = item.get("expected_bytes")
    if destination.exists() and (expected_size is None or destination.stat().st_size >= expected_size):
        valid, dims = valid_2d_mrc(destination)
        if valid:
            return {"status": "reused", "path": str(destination), "bytes": destination.stat().st_size, "shape": dims}

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists() and expected_size is not None and partial.stat().st_size >= expected_size:
        valid, dims = valid_2d_mrc(partial)
        if valid:
            actual_size = partial.stat().st_size
            os.replace(partial, destination)
            return {"status": "resumed_complete", "path": str(destination), "bytes": actual_size, "shape": dims}
        partial.unlink()

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": USER_AGENT}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                append = offset > 0 and response.status == 206
                if offset and not append:
                    offset = 0
                response_total = None
                content_range = response.headers.get("Content-Range", "")
                if "/" in content_range:
                    try:
                        response_total = int(content_range.rsplit("/", 1)[1])
                    except ValueError:
                        response_total = None
                elif response.headers.get("Content-Length"):
                    response_total = offset + int(response.headers["Content-Length"])
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)

            actual_size = partial.stat().st_size
            if response_total is not None and actual_size != response_total:
                raise IOError(f"incomplete HTTP transfer: expected {response_total} bytes, got {actual_size}")
            if expected_size is not None and actual_size < expected_size:
                raise IOError(f"download smaller than EMPIAR index estimate: expected at least {expected_size}, got {actual_size}")
            valid, dims = valid_2d_mrc(partial)
            if not valid:
                raise ValueError(f"not a valid single-frame 2D MRC header: {dims}")
            os.replace(partial, destination)
            return {"status": "downloaded", "path": str(destination), "bytes": actual_size, "shape": dims}
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 20))
    return {"status": "failed", "path": str(destination), "error": f"{type(last_error).__name__}: {last_error}"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = read_json(Path(args.plan))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "sample_manifest.json"
    prior = read_json(manifest_path) if manifest_path.exists() else {"datasets": {}}
    dataset_states = prior.get("datasets", {})

    tasks: list[tuple[str, dict[str, Any], Path]] = []
    for dataset in plan["datasets"]:
        dataset_id = str(dataset["dataset_id"])
        if args.dataset_ids and dataset_id not in set(args.dataset_ids.split(",")):
            continue
        for item in dataset["selected_files"]:
            destination = output_root / f"EMPIAR-{dataset_id}" / Path(item["name"]).name
            tasks.append((dataset_id, item, destination))

    results: dict[str, list[dict[str, Any]]] = {key: [] for key in {x[0] for x in tasks}}
    failures: list[dict[str, Any]] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, item, destination, args.retries): (dataset_id, item, destination)
            for dataset_id, item, destination in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            dataset_id, item, destination = futures[future]
            result = future.result()
            results[dataset_id].append({"name": item["name"], "url": item["url"], **result})
            if result["status"] == "failed":
                failures.append({"dataset_id": dataset_id, "name": item["name"], "error": result["error"]})
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"downloaded/verified {completed}/{len(tasks)} files; failures={len(failures)}", flush=True)

    for dataset_id, files in results.items():
        old = dataset_states.get(dataset_id, {})
        dataset_states[dataset_id] = {
            **old,
            "status": "ready" if len(files) == 20 and all(x["status"] != "failed" for x in files) else "incomplete",
            "directory": str(output_root / f"EMPIAR-{dataset_id}"),
            "file_count": len(files),
            "files": sorted(files, key=lambda x: x["name"]),
            "updated_at": utc_now(),
        }

    payload = {
        "batch_id": plan["batch_id"],
        "plan_path": str(Path(args.plan).resolve()),
        "output_root": str(output_root.resolve()),
        "planned_dataset_count": plan["new_dataset_count"],
        "planned_micrograph_count": plan["total_new_micrograph_count"],
        "updated_at": utc_now(),
        "datasets": dataset_states,
        "failures": failures,
    }
    write_json(manifest_path, payload)
    ready = sum(item.get("status") == "ready" for item in dataset_states.values())
    return {
        "status": "completed" if ready == plan["new_dataset_count"] else "incomplete",
        "ready_dataset_count": ready,
        "planned_dataset_count": plan["new_dataset_count"],
        "failure_count": len(failures),
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--dataset-ids", help="Optional comma-separated EMPIAR IDs for a partial run")
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 0:
        parser.error("--workers must be positive and --retries non-negative")
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
