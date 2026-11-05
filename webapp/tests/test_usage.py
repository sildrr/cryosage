import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from http.client import IncompleteRead
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from cryosparc_agent import model_direct_runner as runner
from webapp import usage


def chat(identifier="chat-1", inputs=10, outputs=4):
    return {"id": identifier, "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": inputs, "completion_tokens": outputs,
                      "total_tokens": inputs + outputs,
                      "prompt_tokens_details": {"cached_tokens": 3}}}


def responses(identifier="resp-1"):
    return {"id": identifier, "object": "response", "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}],
            "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
                      "input_tokens_details": {"cached_tokens": 8}}}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def http_response(value):
    stream = io.BytesIO(json.dumps(value).encode())
    stream.status = 200
    return stream


def call(**kwargs):
    return runner.run_openai_compatible_model(
        [{"role": "user", "content": "secret prompt"}],
        api_base=kwargs.pop("api_base", "https://example.invalid/v1"),
        api_key="secret-key", model_name="test", retry_backoff_seconds=0, **kwargs)


def assert_totals(result, calls, inputs, outputs, cached, failures):
    assert {key: result[key] for key in usage._FIELDS} == {
        "api_calls": calls, "input_tokens": inputs, "output_tokens": outputs,
        "total_tokens": inputs + outputs, "cached_tokens": cached, "failed_calls": failures}


def test_http_attempts_include_retry_failures_and_success(tmp_path, monkeypatch):
    ledger = tmp_path / usage.LEDGER_FILENAME
    monkeypatch.setenv("CRYOAGENT_USAGE_LEDGER", str(ledger))
    failure = HTTPError("secret-url", 429, "rate", {}, io.BytesIO(json.dumps({
        "error": {"message": "secret-error"}, "usage": {"input_tokens": 2, "output_tokens": 1}
    }).encode()))
    with patch.object(runner.request, "urlopen", side_effect=[failure, URLError("secret-error"), http_response(chat())]):
        result = call()
    assert result["attempts"] == 3
    totals = usage.task_usage(tmp_path)
    assert_totals(totals, 3, 12, 5, 3, 2)
    assert totals["source"] == "ledger"
    assert totals["calls_without_usage"] == 1
    assert not totals["tokens_complete"]
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len({row["request_id"] for row in rows}) == 3
    assert "secret" not in ledger.read_text()
    assert ledger.stat().st_mode & 0o777 == 0o600
    # Persisted artifacts and their copied summaries never augment a ledger.
    write(tmp_path / "round_01/model_generation.json", result)
    write(tmp_path / "summary.json", result)
    assert usage.task_usage(tmp_path) == totals


@pytest.mark.parametrize("status,body,options", [
    (400, b"prompt_cache_options unsupported", {"prompt_cache_key": "cache"}),
    (403, b"", {"prompt_cache_key": "cache"}),
    (403, b"", {}),
])
def test_cache_fallback_and_bodyless_forbidden_attempts(tmp_path, monkeypatch, status, body, options):
    monkeypatch.setenv("CRYOAGENT_USAGE_LEDGER", str(tmp_path / usage.LEDGER_FILENAME))
    failure = HTTPError("url", status, "error", {}, io.BytesIO(body))
    with patch.object(runner.request, "urlopen", side_effect=[failure, http_response(chat())]):
        call(**options)
    assert_totals(usage.task_usage(tmp_path), 2, 10, 4, 3, 1)


@pytest.mark.parametrize("failure", [
    HTTPError("url", 401, "auth", {}, io.BytesIO(b"private error")),
    TimeoutError("timeout"), IncompleteRead(b"partial"),
])
def test_terminal_failure_is_recorded(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("CRYOAGENT_USAGE_LEDGER", str(tmp_path / usage.LEDGER_FILENAME))
    with patch.object(runner.request, "urlopen", side_effect=failure), pytest.raises(Exception):
        call(max_retries=0)
    assert_totals(usage.task_usage(tmp_path), 1, 0, 0, 0, 1)


def test_responses_success_counted_once_before_normalization(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYOAGENT_USAGE_LEDGER", str(tmp_path / usage.LEDGER_FILENAME))
    with patch.object(runner.request, "urlopen", return_value=http_response(responses())):
        result = call(api_base="https://packyapi.ai/v1")
    write(tmp_path / "round_01/model_generation.json", result)
    assert_totals(usage.task_usage(tmp_path), 1, 20, 5, 8, 0)


def test_responses_failed_status_preserves_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYOAGENT_USAGE_LEDGER", str(tmp_path / usage.LEDGER_FILENAME))
    response = {**responses(), "status": "failed", "error": {"message": "private"}}
    with patch.object(runner.request, "urlopen", return_value=http_response(response)), pytest.raises(ValueError):
        call(api_base="https://packyapi.ai/v1")
    assert_totals(usage.task_usage(tmp_path), 1, 20, 5, 8, 1)


def test_legacy_only_actual_responses_in_planner_executor_kb_and_reviews(tmp_path):
    response = responses()
    normalized = runner.normalize_responses(response)
    generation = {"raw_response": normalized, "provider_raw_response": response,
                  "attempts": 99, "usage": {"total_tokens": 99999},
                  "tool_trace": [{"tool_name": "kb_search", "result": {"hits": []}},
                                 {"model_response": {"raw_response": chat("kb-1")}}]}
    write(tmp_path / "agent/run_1/round_01/model_generation.json", generation)
    write(tmp_path / "agent/run_1/round_02/model_generation.json", generation)
    write(tmp_path / "agent/run_1/round_01/planner_trace.json", {
        "initial_response": {"raw_response": chat("plan-initial")},
        "observation_response": {"raw_response": chat("plan-observe")}})
    write(tmp_path / "agent/run_1/extract_reviews/J1/pass_1/model_response.json", {"raw_response": chat("review")})
    write(tmp_path / "agent/run_1/summary.json", generation)
    write(tmp_path / "agent/run_1/round_01/kb_tool_calls.json", generation)
    write(tmp_path / "unrelated/model_response.json", {"raw_response": chat("unrelated")})
    result = usage.task_usage(tmp_path)
    assert_totals(result, 5, 60, 21, 20, 0)
    assert result["source"] == "legacy"
    assert not result["api_calls_complete"]
    assert not result["failed_calls_complete"]
    assert "Recorded responses only" in result["note"]


def test_legacy_no_id_aliases_are_not_two_calls_but_distinct_turns_are(tmp_path):
    response = chat(None)
    write(tmp_path / "round_01/model_generation.json", {"raw_response": response, "provider_raw_response": response})
    write(tmp_path / "round_02/model_generation.json", {"raw_response": response})
    assert_totals(usage.task_usage(tmp_path), 2, 20, 8, 6, 0)


def test_legacy_cache_uses_stat_and_refreshes_changes(tmp_path):
    path = tmp_path / "round_01/model_generation.json"
    write(path, {"raw_response": chat()})
    usage.task_usage(tmp_path)
    with patch.object(usage.json, "load", side_effect=AssertionError("reread")):
        assert_totals(usage.task_usage(tmp_path), 1, 10, 4, 3, 0)
    write(path, {"raw_response": chat(inputs=100)})
    assert_totals(usage.task_usage(tmp_path), 1, 100, 4, 3, 0)
    path.unlink()
    assert usage.task_usage(tmp_path)["api_calls"] == 0


def test_ledger_incremental_dedup_partial_lines_and_replacement(tmp_path, monkeypatch):
    path = tmp_path / usage.LEDGER_FILENAME
    monkeypatch.setenv("CRYOAGENT_USAGE_LEDGER", str(path))
    runner._record_api_attempt(json.dumps(chat()), 200, False)
    assert_totals(usage.task_usage(tmp_path), 1, 10, 4, 3, 0)
    line = path.read_text()
    row = json.loads(line)
    row["request_id"] = "another-attempt"
    extra = json.dumps(row) + "\n"
    with path.open("a") as stream:
        stream.write(line + extra[:20])
    assert_totals(usage.task_usage(tmp_path), 1, 10, 4, 3, 0)
    with path.open("a") as stream:
        stream.write(extra[20:])
    assert_totals(usage.task_usage(tmp_path), 2, 20, 8, 6, 0)
    # Same provider response ID on a distinct HTTP attempt still counts.
    row["request_id"] = "third-attempt"
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    assert_totals(usage.task_usage(tmp_path), 3, 30, 12, 9, 0)
    replacement = tmp_path / "replacement"
    replacement.write_text(line)
    os.replace(replacement, path)
    assert_totals(usage.task_usage(tmp_path), 1, 10, 4, 3, 0)


def test_concurrent_appends_are_complete_and_private(tmp_path, monkeypatch):
    path = tmp_path / usage.LEDGER_FILENAME
    path.touch(mode=0o644)
    monkeypatch.setenv("CRYOAGENT_USAGE_LEDGER", str(path))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: runner._record_api_attempt(json.dumps(chat()), 200, False), range(100)))
    assert_totals(usage.task_usage(tmp_path), 100, 1000, 400, 300, 0)
    assert path.stat().st_mode & 0o777 == 0o600


def test_empty_ledger_is_authoritative_and_bad_lines_are_disclosed(tmp_path):
    path = tmp_path / usage.LEDGER_FILENAME
    path.touch()
    write(tmp_path / "round_01/model_generation.json", {"raw_response": chat()})
    assert usage.task_usage(tmp_path)["api_calls"] == 0
    path.write_text("not json\n")
    result = usage.task_usage(tmp_path)
    assert result["unreadable_files"] == 1
    assert not result["api_calls_complete"]


def test_ledger_write_failure_does_not_retry_paid_request(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYOAGENT_USAGE_LEDGER", str(tmp_path / "absent/usage.jsonl"))
    with patch.object(runner.request, "urlopen", return_value=http_response(chat())) as transport:
        with pytest.warns(RuntimeWarning, match="ledger write failed"):
            call()
    assert transport.call_count == 1


def test_unconfigured_ledger_and_preflight_failures_write_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYOAGENT_USAGE_LEDGER", raising=False)
    with patch.object(runner.request, "urlopen", return_value=http_response(chat())):
        call()
    monkeypatch.setenv("CRYOAGENT_USAGE_LEDGER", str(tmp_path / usage.LEDGER_FILENAME))
    with pytest.raises(ValueError):
        call(api_base="")
    assert not (tmp_path / usage.LEDGER_FILENAME).exists()


def test_initialize_new_task_is_exact_and_empty(tmp_path):
    result = usage.initialize_usage(tmp_path)
    assert_totals(result, 0, 0, 0, 0, 0)
    assert not result["has_usage"]
    assert result["source"] == "ledger"
    assert result["report_scope"] == "exact_http_attempts"
    assert result["api_calls_complete"]
    assert (tmp_path / usage.BASELINE_FILENAME).stat().st_mode & 0o777 == 0o600
    assert (tmp_path / usage.LEDGER_FILENAME).stat().st_mode & 0o777 == 0o600


def test_initialize_resume_freezes_legacy_baseline_once(tmp_path, monkeypatch):
    write(tmp_path / "agent/run_old/round_01/model_generation.json", {"raw_response": chat("old")})
    result = usage.initialize_usage(tmp_path)
    assert_totals(result, 1, 10, 4, 3, 0)
    assert result["has_usage"]
    assert result["source"] == "ledger+legacy"
    assert result["report_scope"] == "mixed_recorded_responses_and_http_attempts"
    assert not result["api_calls_complete"]
    assert not result["tokens_complete"]
    baseline = (tmp_path / usage.BASELINE_FILENAME).read_bytes()
    monkeypatch.setenv("CRYOAGENT_USAGE_LEDGER", str(tmp_path / usage.LEDGER_FILENAME))
    runner._record_api_attempt(json.dumps(chat("new")), 200, False)
    runner._record_api_attempt("", 503, True)
    write(tmp_path / "agent/run_new/round_01/model_generation.json", {"raw_response": chat("new")})
    assert_totals(usage.task_usage(tmp_path), 3, 20, 8, 6, 1)
    assert_totals(usage.initialize_usage(tmp_path), 3, 20, 8, 6, 1)
    assert (tmp_path / usage.BASELINE_FILENAME).read_bytes() == baseline
    # Polling neither scans artifact history nor reparses an unchanged baseline.
    with patch.object(usage.os, "walk", side_effect=AssertionError("history scan")):
        with patch.object(usage.json, "load", side_effect=AssertionError("baseline reread")):
            assert_totals(usage.task_usage(tmp_path), 3, 20, 8, 6, 1)


def test_initialize_reuses_snapshot_after_interrupted_initialization(tmp_path):
    write(tmp_path / "round_01/model_generation.json", {"raw_response": chat("old")})
    usage.initialize_usage(tmp_path)
    (tmp_path / usage.LEDGER_FILENAME).unlink()
    write(tmp_path / "round_02/model_generation.json", {"raw_response": chat("not-in-baseline")})
    assert_totals(usage.initialize_usage(tmp_path), 1, 10, 4, 3, 0)


def test_unchanged_ledger_is_not_reparsed_and_append_reads_only_new_rows(tmp_path, monkeypatch):
    usage.initialize_usage(tmp_path)
    monkeypatch.setenv("CRYOAGENT_USAGE_LEDGER", str(tmp_path / usage.LEDGER_FILENAME))
    runner._record_api_attempt(json.dumps(chat()), 200, False)
    usage.task_usage(tmp_path)
    with patch.object(usage.json, "loads", side_effect=AssertionError("ledger reread")):
        assert_totals(usage.task_usage(tmp_path), 1, 10, 4, 3, 0)
    runner._record_api_attempt(json.dumps(chat("new")), 200, False)
    with patch.object(usage.json, "loads", wraps=json.loads) as parse:
        assert_totals(usage.task_usage(tmp_path), 2, 20, 8, 6, 0)
    assert parse.call_count == 1
