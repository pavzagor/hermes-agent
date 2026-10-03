"""Stop-hook source provenance survives injected context and repeated checks."""
from copy import deepcopy
from unittest.mock import patch

from tests.agent.test_verification_continuation_budget import agent, _response


def test_stop_hook_receives_original_request_not_host_envelope(agent, monkeypatch):
    agent.max_iterations = 2
    agent.iteration_budget.max_total = 2
    replies = iter([_response("First draft."), _response("Second draft.")])
    served_requests = []
    def serve(kwargs):
        served_requests.append(str(kwargs))
        return next(replies)
    agent._interruptible_api_call = serve
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    seen = []

    def hook(name, **kwargs):
        if name != "pre_verify":
            return []
        seen.append(kwargs)
        return [{"action": "continue", "message": "Check again."}] if len(seen) == 1 else []

    authored = "Explain the literal <memory-context> tag and preserve this constraint."
    serving = authored + "\nHOST-ONLY ENVELOPE"
    with (
        patch("hermes_cli.plugins.has_hook", side_effect=lambda name: name == "pre_verify"),
        patch("hermes_cli.plugins.invoke_hook", side_effect=hook),
        patch("agent.verify_hooks.pre_verify_every_turn", return_value=True),
    ):
        result = agent.run_conversation(serving, persist_user_message=authored)
    assert len(seen) == 2
    assert [s.get("original_user_message") for s in seen] == [authored, authored]
    assert seen[0].get("turn_id") == seen[1].get("turn_id") and seen[0].get("turn_id")
    assert [s["attempt"] for s in seen] == [0, 1]
    assert "HOST-ONLY ENVELOPE" in served_requests[0]
    assert seen[0].get("current_turn_user_index") == 0
    assert agent._cached_system_prompt == "stable test prompt"
    assert result["final_response"] == "Second draft."


def test_next_turn_replaces_original_request_provenance(agent, monkeypatch):
    agent._interruptible_api_call = lambda _: _response("Answered.")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    seen = []
    def hook(name, **kwargs):
        if name == "pre_verify":
            seen.append(kwargs)
        return []
    with (
        patch("hermes_cli.plugins.has_hook", side_effect=lambda name: name == "pre_verify"),
        patch("hermes_cli.plugins.invoke_hook", side_effect=hook),
        patch("agent.verify_hooks.pre_verify_every_turn", return_value=True),
    ):
        agent.run_conversation("First envelope.", persist_user_message="First request.")
        agent.run_conversation("Second request.")
    assert [s.get("original_user_message") for s in seen] == ["First request.", "Second request."]


def test_multimodal_original_request_does_not_alias_host_context_parts(agent, monkeypatch):
    monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    monkeypatch.setattr("agent.turn_context._collect_pre_llm_call_context",
                        lambda *a, **kw: "[Automatically loaded skills, full content.]\nPRIVATE_HOST_SKILL")
    authored = [{"type": "text", "text": "Explain the image and literal <memory-context> tag."},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]
    original = deepcopy(authored)
    seen, served = [], []

    def serve(kwargs):
        served.append(deepcopy(kwargs))
        return _response("Answered.")

    def hook(name, **kwargs):
        if name == "pre_verify":
            seen.append(deepcopy(kwargs))
        return []

    agent._interruptible_api_call = serve
    with (
        patch("hermes_cli.plugins.has_hook", side_effect=lambda name: name == "pre_verify"),
        patch("hermes_cli.plugins.invoke_hook", side_effect=hook),
        patch("agent.verify_hooks.pre_verify_every_turn", return_value=True),
    ):
        agent.run_conversation(authored)
    assert len(seen) == 1
    assert seen[0]["original_user_message"] == original
    assert agent._turn_original_user_message == original
    assert "PRIVATE_HOST_SKILL" in str(served)  # Serving context still reaches the main model.
    assert served[0]["messages"][-1]["content"] == [*original,
        {"type": "text", "text": "[Automatically loaded skills, full content.]\nPRIVATE_HOST_SKILL"}]
