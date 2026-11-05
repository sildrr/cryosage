import copy

import pytest

from api_responses_adapter import normalize_responses, responses_payload


def test_images_cache_and_tool_schemas_translate_without_mutating_messages():
    chat = {
        "model": "gpt-6.1-sol", "max_tokens": 1024, "temperature": 0,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "rules",
                "prompt_cache_breakpoint": {"mode": "explicit"}}]},
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA==", "detail": "high"}},
            ]},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "tool_choice": "auto", "prompt_cache_key": "stable", "prompt_cache_options": {"mode": "explicit"},
    }
    before = copy.deepcopy(chat)
    result = responses_payload(chat)
    assert chat == before
    assert result["input"][1]["content"][1] == {
        "type": "input_image", "image_url": "data:image/png;base64,eA==", "detail": "high",
    }
    assert result["tools"][0]["name"] == "lookup"
    assert result["prompt_cache_key"] == "stable"
    assert "prompt_cache_options" not in result
    assert "prompt_cache_breakpoint" not in str(result)
    assert result["store"] is False
    assert "temperature" not in result


def test_tool_roundtrip_preserves_reasoning_and_exact_call_ids():
    output = [
        {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "opaque"},
        {"id": "fc_1", "type": "function_call", "call_id": "call_abc", "name": "lookup", "arguments": '{"id":1}'},
    ]
    raw = {"status": "completed", "output": output,
           "usage": {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125}}
    normalized = normalize_responses(raw)
    choice = normalized["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["id"] == "call_abc"
    assert normalized["usage"]["prompt_tokens"] == 100
    request = responses_payload({"model": "gpt-6.1-sol", "max_tokens": 1024, "messages": [
        {"role": "user", "content": "lookup"}, choice["message"],
        {"role": "tool", "tool_call_id": "call_abc", "content": '{"value":42}', "name": "lookup"},
    ]})
    assert request["input"][1:3] == output
    assert request["input"][3] == {
        "type": "function_call_output", "call_id": "call_abc", "output": '{"value":42}',
    }
    assert raw["output"] == output


def test_plain_assistant_and_legacy_chat_tool_messages_are_supported():
    result = responses_payload({"model": "m", "max_tokens": 10, "messages": [
        {"role": "assistant", "content": "checking", "tool_calls": [
            {"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
    ]})
    assert result["input"][0] == {"role": "assistant", "content": "checking"}
    assert result["input"][1]["type"] == "function_call"
    assert result["input"][2]["type"] == "function_call_output"


def test_length_and_refusal_are_not_reported_as_success():
    result = normalize_responses({"status": "incomplete", "output": []})
    assert result["choices"][0]["finish_reason"] == "length"
    with pytest.raises(ValueError, match="refused"):
        normalize_responses({"output": [{"type": "message", "content": [
            {"type": "refusal", "refusal": "no"},
        ]}]})
    with pytest.raises(ValueError, match="failed"):
        normalize_responses({"status": "failed", "output": [], "error": {"message": "failed"}})


def test_unsupported_evidence_is_not_silently_removed():
    with pytest.raises(ValueError, match="no evidence was dropped"):
        responses_payload({"model": "m", "max_tokens": 10, "messages": [
            {"role": "user", "content": [{"type": "input_audio", "data": "unknown"}]},
        ]})
