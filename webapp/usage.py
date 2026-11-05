"""Workflow API accounting, independent of the web server and worker.

Parent workers must export CRYOAGENT_USAGE_LEDGER=str(root / LEDGER_FILENAME)
to every controller/model subprocess. An existing ledger is authoritative, even
when empty: legacy artifacts cannot reconstruct HTTP failures or unsaved turns.
"""
from __future__ import annotations

from collections import OrderedDict
import fcntl
import json
import os
from pathlib import Path
import threading
import tempfile

LEDGER_FILENAME = "api_usage.jsonl"
BASELINE_FILENAME = "usage_baseline.json"
_FIELDS = ("api_calls", "input_tokens", "output_tokens", "total_tokens", "cached_tokens", "failed_calls")
_LOCK = threading.RLock()
_LEDGERS: OrderedDict = OrderedDict()
_FILES: OrderedDict = OrderedDict()
_BASELINES: OrderedDict = OrderedDict()


def _number(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _tokens(usage):
    inputs = _number(usage.get("input_tokens", usage.get("prompt_tokens")))
    outputs = _number(usage.get("output_tokens", usage.get("completion_tokens")))
    details = usage.get("input_tokens_details", usage.get("prompt_tokens_details")) or {}
    return {
        "input_tokens": inputs, "output_tokens": outputs,
        "total_tokens": _number(usage.get("total_tokens")) or inputs + outputs,
        "cached_tokens": _number(details.get("cached_tokens")) if isinstance(details, dict) else 0,
    }


def _remember(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def _empty():
    return dict.fromkeys(_FIELDS, 0)


def _ledger_usage(path):
    with path.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        stat = os.fstat(stream.fileno())
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        state = _LEDGERS.get(str(path))
        if state and state["signature"] == signature:
            return dict(state["totals"]), state["missing"], state["invalid"] + state["pending"]
        if not state or state["signature"][:2] != signature[:2] or stat.st_size <= state["signature"][2]:
            state = {"offset": 0, "seen": set(), "totals": _empty(), "missing": 0, "invalid": 0}
        stream.seek(state["offset"])
        state["pending"] = 0
        for line in stream:
            if not line.endswith(b"\n"):
                state["pending"] = 1
                break  # Retry an incomplete final append on the next poll.
            state["offset"] += len(line)
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("request_id"), str):
                    raise ValueError("missing request ID")
            except (ValueError, UnicodeError):
                state["invalid"] += 1
                continue
            # A request ID identifies an HTTP attempt, not a provider response.
            if row["request_id"] in state["seen"]:
                continue
            state["seen"].add(row["request_id"])
            totals = state["totals"]
            totals["api_calls"] += 1
            totals["failed_calls"] += int(bool(row.get("failed")))
            for key in _FIELDS[1:5]:
                totals[key] += _number(row.get(key))
            state["missing"] += int(not row.get("usage_recorded"))
        state["signature"] = signature
        _remember(_LEDGERS, str(path), state, 128)
        return dict(state["totals"]), state["missing"], state["invalid"] + state["pending"]


def _responses(value):
    """Visit actual response objects, never token summaries or request messages."""
    if isinstance(value, list):
        for item in value:
            yield from _responses(item)
    elif isinstance(value, dict):
        if (isinstance(value.get("choices"), list) or value.get("object") == "response"
                or isinstance(value.get("output"), list) and "usage" in value):
            usage = value.get("usage")
            yield (value.get("id"), _tokens(usage if isinstance(usage, dict) else {}),
                   bool(value.get("error")) or value.get("status") in {"failed", "cancelled"}, bool(usage))
            return
        # Normalized Chat and provider Responses payloads represent the same call.
        raw = value.get("provider_raw_response")
        if not isinstance(raw, dict):
            raw = value.get("raw_response")
        if isinstance(raw, dict):
            yield from _responses(raw)
        for key, item in value.items():
            if key not in {"raw_response", "provider_raw_response", "request_payload", "messages",
                           "messages_after_tools", "assistant_message", "summary"}:
                yield from _responses(item)


def _legacy_file(path):
    stat = path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _FILES.get(str(path))
    if cached and cached[0] == signature:
        _FILES.move_to_end(str(path))
        return cached[1]
    try:
        with path.open(encoding="utf-8") as stream:
            rows = list(_responses(json.load(stream)))
    except (ValueError, UnicodeError):
        # A file may be mid-write; do not cache a failed parse.
        return [], 1
    return _remember(_FILES, str(path), (signature, (rows, 0)), 2048)[1]


def _baseline(root):
    path = root / BASELINE_FILENAME
    try:
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = _BASELINES.get(str(path))
        if cached and cached[0] == signature:
            return cached[1]
        with path.open(encoding="utf-8") as stream:
            snapshot = json.load(stream)
        if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
            raise ValueError("invalid baseline")
        return _remember(_BASELINES, str(path), (signature, (snapshot, 0)), 128)[1]
    except FileNotFoundError:
        return {}, 0
    except (OSError, ValueError, UnicodeError):
        return {}, 1


def initialize_usage(root):
    """Call before launching the worker's first/resumed model subprocess.

    Freeze pre-ledger history exactly once, then create the ledger. A private
    initializer lock serializes workers; atomic baseline replacement precedes
    ledger creation, so a crash between them can safely be retried. Do not call
    this concurrently with an already-running, uninitialized API producer.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        fd = os.open(root / ".usage_init.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = root / LEDGER_FILENAME
            if not ledger.exists():
                baseline = root / BASELINE_FILENAME
                if not baseline.exists():
                    snapshot = {"version": 1, **task_usage(root)}
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                                         prefix=".usage_baseline-", delete=False) as stream:
                            temporary = Path(stream.name)
                            json.dump(snapshot, stream, separators=(",", ":"))
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, baseline)
                    finally:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
                try:
                    fd = os.open(ledger, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                except FileExistsError:
                    pass
                else:
                    os.close(fd)
            return task_usage(root)


def task_usage(root):
    """Return JSON-safe totals for a workflow directory (safe for frequent polls).

    Tokens include cached input; cached_tokens is a subset, not an extra charge.
    Ledger counts are actual HTTP attempts. Legacy counts are only distinct
    recorded responses; retry metadata is deliberately not treated as evidence.
    """
    root = Path(root).resolve()
    with _LOCK:
        ledger = root / LEDGER_FILENAME
        try:
            totals, missing, invalid = _ledger_usage(ledger)
            baseline, baseline_errors = _baseline(root)
            for key in _FIELDS:
                totals[key] += _number(baseline.get(key))
            missing += _number(baseline.get("calls_without_usage"))
            invalid += _number(baseline.get("unreadable_files")) + baseline_errors
            mixed = bool(baseline.get("has_usage") or baseline.get("unreadable_files") or baseline_errors)
            return {**totals, "source": "ledger+legacy" if mixed else "ledger",
                    "has_usage": bool(totals["api_calls"]),
                    "report_scope": "mixed_recorded_responses_and_http_attempts" if mixed else "exact_http_attempts",
                    "api_calls_complete": not (invalid or mixed),
                    "failed_calls_complete": not (invalid or mixed), "tokens_complete": not (missing or invalid or mixed),
                    "calls_without_usage": missing, "unreadable_files": invalid,
                    "note": ("Legacy recorded-response baseline plus new HTTP attempts; historical failures/retries may be missing."
                             if mixed else "HTTP attempts recorded since ledger tracking began; unreported tokens are unknown.")}
        except FileNotFoundError:
            pass
        except OSError:
            return {**_empty(), "source": "ledger", "has_usage": False,
                    "report_scope": "exact_http_attempts", "api_calls_complete": False,
                    "failed_calls_complete": False, "tokens_complete": False,
                    "calls_without_usage": 0, "unreadable_files": 1, "note": "Usage ledger is unreadable."}
        totals, seen, missing, unreadable = _empty(), set(), 0, 0
        for directory, _, names in os.walk(root):
            for name in sorted(names):
                path = Path(directory) / name
                if name not in {"model_generation.json", "planner_trace.json"} and not (
                    name == "model_response.json" and "extract_reviews" in path.relative_to(root).parts
                ):
                    continue
                try:
                    rows, errors = _legacy_file(path)
                except OSError:
                    unreadable += 1
                    continue
                unreadable += errors
                for response_id, tokens, failed, recorded in rows:
                    if isinstance(response_id, str) and response_id:
                        if response_id in seen:
                            continue
                        seen.add(response_id)
                    totals["api_calls"] += 1
                    totals["failed_calls"] += int(failed)
                    missing += int(not recorded)
                    for key, count in tokens.items():
                        totals[key] += count
        return {**totals, "source": "legacy", "has_usage": bool(totals["api_calls"]),
                "report_scope": "approximate_recorded_responses", "api_calls_complete": False,
                "failed_calls_complete": False, "tokens_complete": False,
                "calls_without_usage": missing, "unreadable_files": unreadable,
                "note": "Recorded responses only; HTTP failures, retries, and unsaved KB turns may be missing."}
