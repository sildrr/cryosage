"""Translate the Agent's chat/tool contract to stateless Responses requests."""

import copy


def responses_payload(chat):
    inputs = []
    for message in chat["messages"]:
        role = message["role"]
        if role == "tool":
            inputs.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                           "output": message.get("content") or ""})
            continue
        if role == "assistant" and message.get("_responses_output"):
            # Preserve reasoning/encrypted state and exact call IDs across tool turns.
            inputs.extend(copy.deepcopy(message["_responses_output"]))
            continue
        content = message.get("content")
        if content:
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"type": "output_text" if role == "assistant" else "input_text",
                                      "text": part["text"]})
                    elif part.get("type") == "image_url" and role == "user":
                        image = part["image_url"]
                        parts.append({"type": "input_image", "image_url": image["url"],
                                      "detail": image.get("detail", "auto")})
                    else:
                        raise ValueError("Unsupported content in Responses input; no evidence was dropped")
                content = parts
            inputs.append({"role": role, "content": content})
        for call in message.get("tool_calls") or []:
            inputs.append({"type": "function_call", "call_id": call["id"],
                           "name": call["function"]["name"],
                           "arguments": call["function"]["arguments"]})
    payload = {
        "model": chat["model"], "input": inputs,
        "max_output_tokens": chat["max_tokens"], "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if chat.get("tools"):
        tools = []
        for tool in chat["tools"]:
            if tool.get("type") != "function":
                raise ValueError("Only function tools are supported by this Agent adapter")
            tools.append({"type": "function", **tool["function"]})
        payload["tools"] = tools
    if chat.get("tool_choice"):
        choice = chat["tool_choice"]
        if isinstance(choice, dict) and choice.get("type") == "function":
            choice = {"type": "function", "name": choice["function"]["name"]}
        payload["tool_choice"] = choice
    if chat.get("prompt_cache_key"):
        payload["prompt_cache_key"] = chat["prompt_cache_key"]
    # Explicit cache breakpoints/options are ofox extensions, not Responses fields.
    return payload


def normalize_responses(response):
    if response.get("error") or response.get("status") in {"failed", "cancelled"}:
        raise ValueError(f"Responses request failed: {response.get('error') or response.get('status')}")
    if not isinstance(response.get("output"), list):
        raise ValueError("Responses response did not contain an output list")
    texts, calls = [], []
    for item in response["output"]:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(part["text"])
                elif part.get("type") == "refusal":
                    raise ValueError(f"Responses model refused: {part.get('refusal', '')}")
        elif item.get("type") == "function_call":
            calls.append({"id": item["call_id"], "type": "function", "function": {
                "name": item["name"], "arguments": item["arguments"],
            }})
    message = {"role": "assistant", "content": "\n".join(texts) or None,
               "_responses_output": copy.deepcopy(response["output"])}
    if calls:
        message["tool_calls"] = calls
    incomplete = response.get("status") == "incomplete"
    reason = "length" if incomplete else ("tool_calls" if calls else "stop")
    usage = dict(response.get("usage") or {})
    usage.update(prompt_tokens=usage.get("input_tokens", 0),
                 completion_tokens=usage.get("output_tokens", 0),
                 prompt_tokens_details=usage.get("input_tokens_details") or {},
                 completion_tokens_details=usage.get("output_tokens_details") or {})
    normalized = dict(response)
    normalized.update(choices=[{"index": 0, "message": message, "finish_reason": reason}], usage=usage)
    return normalized
