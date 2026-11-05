# Tests direct-model JSON parsing helpers without loading the model.
import io
import json
import ssl
import unittest
from unittest import mock
from urllib.error import HTTPError

from model_direct_runner import (
    build_workflow_decision_prompt,
    normalize_optional_path,
    parse_model_decision_text,
    run_openai_compatible_model,
)


class ModelDirectRunnerTests(unittest.TestCase):
    def test_transient_tls_record_failure_retries_without_changing_payload(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'
        with mock.patch("model_direct_runner.request.urlopen", side_effect=[
            ssl.SSLError("[SSL] record layer failure (_ssl.c:2590)"), response,
        ]) as call, mock.patch("model_direct_runner.time.sleep") as sleep:
            result = run_openai_compatible_model([], "https://example.test/v1", "key", "model",
                                                 max_retries=1, retry_backoff_seconds=1)
        self.assertEqual(result["raw_text"], "OK")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args_list[0].args[0].data, call.call_args_list[1].args[0].data)
        sleep.assert_called_once()

    def test_certificate_failure_is_not_retried_or_bypassed(self):
        with mock.patch("model_direct_runner.request.urlopen", side_effect=
                        ssl.SSLCertVerificationError(1, "certificate verify failed")) as call:
            with self.assertRaisesRegex(RuntimeError, "certificate verify failed"):
                run_openai_compatible_model([], "https://example.test/v1", "key", "model", max_retries=3)
        self.assertEqual(call.call_count, 1)

    def test_transient_tls_retry_budget_is_bounded(self):
        with mock.patch("model_direct_runner.request.urlopen", side_effect=
                        ssl.SSLError("record layer failure")) as call, mock.patch("model_direct_runner.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "OpenAI-compatible request failed to reach"):
                run_openai_compatible_model([], "https://example.test/v1", "key", "model", max_retries=1)
        self.assertEqual(call.call_count, 2)

    def _provider_probe(self, base, model, provider=None):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("CRYOAGENT_OFOX_PROVIDER_TYPE", None)
            if provider is not None:
                os.environ["CRYOAGENT_OFOX_PROVIDER_TYPE"] = provider
            with mock.patch("model_direct_runner.request.urlopen", return_value=response) as call:
                result = run_openai_compatible_model([], base, "key", model, max_retries=0)
                return result, call.call_args.args[0]

    def test_ofox_gpt6_models_pin_healthy_route_without_changing_payload(self):
        for model in ("openai/gpt-6.1-sol", "gpt-6.1-sol", "openai/gpt-6-luna", "gpt-6-luna"):
            result, req = self._provider_probe("https://api.ofox.ai/v1", model)
            self.assertEqual(req.get_header("X-ofoxai-provider-type"), "openai")
            self.assertEqual(json.loads(req.data)["model"], model)
            self.assertEqual(result["provider_type"], "openai")
            self.assertNotIn("provider", json.loads(req.data))

    def test_ofox_route_can_be_explicitly_disabled(self):
        result, req = self._provider_probe("https://api.ofox.ai/v1", "openai/gpt-6.1-sol", "auto")
        self.assertIsNone(req.get_header("X-ofoxai-provider-type"))
        self.assertIsNone(result["provider_type"])

    def test_ofox_route_can_be_explicitly_selected(self):
        result, req = self._provider_probe("https://api.ofox.ai/v1", "openai/gpt-6.1-sol", "azure_foundry")
        self.assertEqual(req.get_header("X-ofoxai-provider-type"), "azure_foundry")
        self.assertEqual(result["provider_type"], "azure_foundry")

    def test_other_ofox_models_keep_existing_route(self):
        _, req = self._provider_probe("https://api.ofox.ai/v1", "openai/gpt-5.6-sol")
        self.assertIsNone(req.get_header("X-ofoxai-provider-type"))

    def test_route_setting_is_not_forwarded_to_other_hosts(self):
        _, req = self._provider_probe("https://api.ofox.ai.example.test/v1", "openai/gpt-6.1-sol", "openai")
        self.assertIsNone(req.get_header("X-ofoxai-provider-type"))

    def test_invalid_provider_does_not_send_request(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self._provider_probe("https://api.ofox.ai/v1", "openai/gpt-6.1-sol", "not_a_provider")

    def test_packy_uses_responses_and_preserves_tools(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"OK"}]}]}'
        for base in ("https://www.packyapi.ai/v1", "https://packyapi.ai/v1"):
            with self.subTest(base=base), mock.patch(
                "model_direct_runner.request.urlopen", return_value=response,
            ) as urlopen:
                result = run_openai_compatible_model(
                    [{"role": "user", "content": "test"}], base, "key", "gpt-6.1-sol",
                    max_new_tokens=1024, temperature=0,
                    tools=[{"type": "function", "function": {"name": "probe"}}],
                    tool_choice="auto",
                )
                sent = json.loads(urlopen.call_args.args[0].data)
                self.assertEqual(sent["max_output_tokens"], 1024)
                self.assertNotIn("max_tokens", sent)
                self.assertNotIn("temperature", sent)
                self.assertNotIn("reasoning_effort", sent)
                self.assertNotIn("reasoning", sent)
                self.assertEqual(sent["tool_choice"], "auto")
                self.assertEqual(sent["tools"][0]["name"], "probe")
                self.assertEqual(result["request_profile"], "packy_responses")
                self.assertTrue(urlopen.call_args.args[0].full_url.endswith("/responses"))
                self.assertEqual(result["raw_text"], "OK")

    def test_other_hosts_keep_existing_parameters(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'
        for base in ("https://api.ofox.ai/v1", "https://packyapi.ai.example.test/v1"):
            with self.subTest(base=base), mock.patch(
                "model_direct_runner.request.urlopen", return_value=response,
            ) as urlopen:
                result = run_openai_compatible_model(
                    [], base, "key", "model", max_new_tokens=123, temperature=0.2,
                )
                sent = json.loads(urlopen.call_args.args[0].data)
                self.assertEqual(sent["max_tokens"], 123)
                self.assertEqual(sent["temperature"], 0.2)
                self.assertNotIn("max_completion_tokens", sent)
                self.assertEqual(result["request_profile"], "openai_compatible")

    def test_packy_without_tools_does_not_force_reasoning_off(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"status":"completed","output":[]}'
        with mock.patch("model_direct_runner.request.urlopen", return_value=response) as urlopen:
            run_openai_compatible_model([], "https://www.packyapi.ai/v1", "key", "gpt-6.1-sol")
        self.assertNotIn("reasoning_effort", json.loads(urlopen.call_args.args[0].data))

    def test_parse_plain_json(self):
        result = parse_model_decision_text(
            '{"schema_version":"2.0","decision_type":"stop","reason":"done"}'
        )

        self.assertEqual(result["decision_type"], "stop")

    def test_parse_markdown_wrapped_json(self):
        result = parse_model_decision_text(
            '```json\n{"schema_version":"2.0","decision_type":"forward"}\n```'
        )

        self.assertEqual(result["decision_type"], "forward")

    def test_parse_output_after_thinking_text(self):
        result = parse_model_decision_text(
            "/think hidden notes\n"
            '{"schema_version":"2.0","decision_type":"stop","reason":"review"}'
        )

        self.assertEqual(result["reason"], "review")

    def test_prompt_contains_v2_context_and_contract(self):
        messages = build_workflow_decision_prompt(
            {
                "schema_version": "2.0",
                "task_type": "workflow_decision",
                "dataset_info": {},
                "current_state": {"last_node_id": "J8"},
            }
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("output_contract", messages[1]["content"])
        self.assertIn("J8", messages[1]["content"])

    def test_empty_adapter_path_is_normalized(self):
        self.assertIsNone(normalize_optional_path(""))
        self.assertIsNone(normalize_optional_path("none"))
        self.assertEqual(normalize_optional_path("/tmp/adapter"), "/tmp/adapter")

    def test_api_retries_transient_errors(self):
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'
        errors = [
            HTTPError("https://example.test", 500, "server", {}, io.BytesIO(b"{}")),
            HTTPError("https://example.test", 429, "busy", {}, io.BytesIO(b"{}")),
            response,
        ]
        with mock.patch("model_direct_runner.request.urlopen", side_effect=errors) as urlopen:
            with mock.patch("model_direct_runner.time.sleep") as sleep:
                result = run_openai_compatible_model(
                    [{"role": "user", "content": "test"}],
                    "https://example.test/v1",
                    "key",
                    "model",
                    max_retries=2,
                    retry_backoff_seconds=2,
                )
        self.assertEqual(result["raw_text"], "OK")
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def test_api_drops_unsupported_cache_extensions(self):
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'
        unsupported = HTTPError(
            "https://example.test", 500, "server", {},
            io.BytesIO(b'{"error":{"message":"prompt_cache_breakpoint is not supported"}}'),
        )
        with mock.patch(
            "model_direct_runner.request.urlopen", side_effect=[unsupported, response]
        ) as urlopen:
            result = run_openai_compatible_model(
                [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "test",
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }],
                }],
                "https://example.test/v1",
                "key",
                "model",
                prompt_cache_key="cache-key",
                prompt_cache_options={"mode": "explicit"},
                max_retries=0,
            )
        self.assertEqual(result["raw_text"], "OK")
        self.assertTrue(result["cache_fallback_used"])
        self.assertEqual(urlopen.call_count, 2)
        first_payload = json.loads(urlopen.call_args_list[0].args[0].data)
        second_payload = json.loads(urlopen.call_args_list[1].args[0].data)
        self.assertIn("prompt_cache_key", first_payload)
        self.assertNotIn("prompt_cache_key", second_payload)
        self.assertNotIn(
            "prompt_cache_breakpoint",
            json.dumps(second_payload["messages"]),
        )


if __name__ == "__main__":
    unittest.main()
