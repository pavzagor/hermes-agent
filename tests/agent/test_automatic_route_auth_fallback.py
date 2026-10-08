"""Automatic provider routing must not discard the configured safety net."""
import json
import sys
import httpx
from openai import OpenAI
import pytest

from run_agent import AIAgent


NATIVE_PROVIDER = "claude-subscription-directsdk-experimental"
LOGIN_ERROR = "Native API error: Not logged in · Please run /login"


@pytest.mark.parametrize("entry_path", ["explicit", "automatic_routing", "automatic_restore"])
def test_automatic_route_login_failure_reaches_same_provider_model_backup(tmp_path, monkeypatch, entry_path):
    """Real routing, resolution and turn recovery; only credentials/provider I/O are fake.

    Claude is already the first configured backup. Routing to it must still allow
    Sol then Luna on the original Codex provider, without rebuilding the prefix.
    """
    from agent import auxiliary_client, process_bootstrap
    from hermes_cli import lifecycle

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "model-providers" / "native-test"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(
        'from providers import register_provider\n'
        'from providers.base import ProviderProfile\n'
        'from unittest.mock import MagicMock\n'
        'class Native(ProviderProfile):\n'
        '    attempts = 0\n'
        '    def create_client(self, **kwargs):\n'
        '        client = MagicMock()\n'
        '        client.base_url = self.base_url\n'
        '        client.api_key = "process-placeholder"\n'
        '        def create(**request):\n'
        '            self.attempts += 1\n'
        f'            raise RuntimeError({LOGIN_ERROR!r})\n'
        '        client.chat.completions.create.side_effect = create\n'
        '        return client\n'
        f'register_provider(Native(name={NATIVE_PROVIDER!r}, auth_type="external_process",\n'
        f'    process_command={sys.executable!r}, base_url="process://{NATIVE_PROVIDER}",\n'
        '    supports_model_listing=False, supports_health_check=False))\n',
        encoding="utf-8",
    )
    chain = [
        {"provider": NATIVE_PROVIDER, "model": "claude-opus-5-5[1m]"},
        # Explicit chat wire keeps this test about routing/failover, not Responses serialization.
        {"provider": "openai-codex", "model": "gpt-6-sol", "api_mode": "chat_completions"},
        {"provider": "openai-codex", "model": "gpt-6-luna", "api_mode": "chat_completions"},
    ]
    (tmp_path / "config.yaml").write_text(json.dumps({
        "fallback_providers": chain,
        "turn_routing": {"enabled": True, "models": {
            NATIVE_PROVIDER: ["claude-opus-5-5[1m]"],
        }},
    }), encoding="utf-8")
    sent = []

    def respond(request):
        body = json.loads(request.content)
        sent.append(body)
        if body["model"] == "gpt-6-sol":
            return httpx.Response(404, json={"error": {"message": "model not found"}})
        assert body["model"] == "gpt-6-luna"
        return httpx.Response(200, json={
            "id": "fallback-ok", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "Recovered on Luna."},
                "finish_reason": "stop"}],
        })

    def sdk_client(**kwargs):
        old_http = kwargs.pop("http_client", None)
        if old_http:
            old_http.close()
        return OpenAI(**{**kwargs, "max_retries": 0},
                      http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(process_bootstrap, "OpenAI", sdk_client)
    monkeypatch.setattr(auxiliary_client, "OpenAI", sdk_client)
    monkeypatch.setattr(auxiliary_client, "_resolve_codex_credential_and_base",
                        lambda: ("fake-codex-token", "https://chatgpt.com/backend-api/codex"))
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda event, **kw: [
        {"route": {"provider": NATIVE_PROVIDER, "model": "claude-opus-5-5[1m]"}}
    ] if event == "pre_llm_call" and entry_path == "automatic_routing" else [])
    from hermes_cli.config import load_config
    from hermes_cli.fallback_config import get_fallback_chain
    from agent import turn_routing
    session_id = "fallback-regression"
    if entry_path == "automatic_restore":
        # Persist via the host's routing API, then recreate the agent from its baseline.
        from types import SimpleNamespace
        prior = SimpleNamespace(session_id=session_id, provider=NATIVE_PROVIDER,
                                model="claude-opus-5-5[1m]")
        assert not turn_routing.apply_route(prior, {
            "provider": NATIVE_PROVIDER, "model": "claude-opus-5-5[1m]",
        }, is_first_turn=True)
    explicit = entry_path == "explicit"
    agent = AIAgent(
        provider=NATIVE_PROVIDER if explicit else "openai-codex",
        model="claude-opus-5-5[1m]" if explicit else "gpt-6-astra",
        api_key="fake-codex-token",
        base_url=f"process://{NATIVE_PROVIDER}" if explicit else "https://chatgpt.com/backend-api/codex",
        api_mode="chat_completions", session_id=session_id,
        fallback_model=get_fallback_chain(load_config()), enabled_toolsets=[], quiet_mode=True,
        skip_context_files=True, skip_memory=True, save_trajectories=False,
    )
    agent._cached_system_prompt = "Stable conversation prefix."
    agent._disable_streaming = True
    agent.compression_enabled = False
    # No delays if the broken base retries its deterministic login failure.
    monkeypatch.setattr("agent.turn_recovery.interruptible_backoff_sleep", lambda *a, **kw: None)
    result = agent.run_conversation("Reply with a brief greeting.")

    assert result["completed"] is True, result
    assert result["final_response"] == "Recovered on Luna."
    assert [request["model"] for request in sent] == ["gpt-6-sol", "gpt-6-luna"]
    from providers import get_provider_profile
    assert get_provider_profile(NATIVE_PROVIDER).attempts == 3
    assert agent._fallback_chain == chain
    assert agent._cached_system_prompt == "Stable conversation prefix."
    assert sent[0]["messages"] == sent[1]["messages"]
    assert [m["role"] for m in result["messages"]] == ["user", "assistant"]
