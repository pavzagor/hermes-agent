"""Scheduler -> real host pin acceptance, with synthetic config and no inference."""
from __future__ import annotations

import json
import socket
from typing import Any

import pytest
import yaml

from agent import turn_routing
from agent.turn_context import _collect_pre_llm_call_context
from cron import scheduler
from hermes_constants import get_hermes_home


@pytest.fixture
def host_config(monkeypatch: Any) -> dict[str, Any]:
    def forbid_network(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Offline host fixtures must not open network connections")
    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    cfg = {
        "model": {"default": "gpt-6-astra", "provider": "openai-codex"},
        "cron": {"model": "gpt-6-astra", "model_provider": "openai-codex"},
        "turn_routing": {"enabled": True, "routes": {
            "sol": {"provider": "openai-codex", "model": "gpt-5.6-sol"}}},
        "reasoning_routing": {"enabled": True},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
    }
    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    return cfg


def construct(job: dict[str, Any], cfg: dict[str, Any], sid: str) -> Any:
    from run_agent import AIAgent
    setup = scheduler._CronAgentSetup(
        model="gpt-6-astra", max_iterations=3,
        runtime={"provider": "openai-codex", "api_key": "synthetic-fixture-key",
                 "base_url": "http://127.0.0.1:1/v1", "api_mode": "chat_completions"},
    )
    agent = scheduler._construct_cron_agent(
        AIAgent, {"id": sid, "enabled_toolsets": ["no_mcp"], **job}, cfg, setup,
        workdir=None, session_id=sid, session_db=None,
    )
    agent._cached_system_prompt = "Frozen synthetic prefix."
    assert agent.tools == []
    return agent


@pytest.mark.parametrize("pin", [
    {"model": "gpt-6-astra"}, {"provider": "openai-codex"},
    {"base_url": "http://127.0.0.1:1/v1"},
    {"model": "gpt-6-astra", "provider": "openai-codex"},
])
def test_real_pinned_host_keeps_skills_and_independent_effort(
    pin: dict[str, str], host_config: dict[str, Any], monkeypatch: Any,
) -> None:
    import hermes_cli.lifecycle
    import tools.skills_tool
    loaded: list[str] = []
    hooks: list[dict[str, Any]] = []

    def classify(hook: str, **kwargs: Any) -> list[dict[str, Any]]:
        assert hook == "pre_llm_call"
        hooks.append(kwargs)
        return [{"route": {"provider": "openai-codex", "model": "gpt-5.6-sol"},
                 "reasoning": {"effort": "low"}, "skills": ["public-skill"]}]

    def skill(name: str, **kwargs: Any) -> str:
        loaded.append(name)
        return json.dumps({"success": True, "content": "FULL PUBLIC SKILL"})

    monkeypatch.setattr(hermes_cli.lifecycle, "invoke_hook", classify)
    monkeypatch.setattr(tools.skills_tool, "skill_view", skill)
    agent = construct(pin, host_config, "cron_public_pin")
    client = agent.client
    try:
        context = _collect_pre_llm_call_context(
            agent, effective_task_id="public", turn_id="public-turn", original_user_message="Read public fixture.",
            messages=[{"role": "user", "content": "Read public fixture."}], conversation_history=[],
        )
        assert agent.model == "gpt-6-astra"
        assert agent.provider == "openai-codex"
        assert agent.reasoning_config == {"enabled": True, "effort": "low"}
        assert agent.reasoning_provenance == "automatic"
        assert hooks[0]["reasoning_pinned"] is False
        assert hooks[0]["platform"] == "cron"
        assert loaded == ["public-skill"] and "FULL PUBLIC SKILL" in context
        assert agent._cached_system_prompt == "Frozen synthetic prefix."
        assert not turn_routing.apply_route(agent, {"model": "gpt-5.6-sol"}, is_first_turn=True)
    finally:
        scheduler._teardown_cron_agent(agent, "public", timeout_seconds=0)
    assert client.is_closed()
    # Durable session identity remains pinned, not a process-wide routing switch.
    assert not turn_routing.routing_enabled(agent.session_id, is_first_turn=True)
    other = construct({}, host_config, "cron_public_unpinned")
    try:
        assert turn_routing.apply_route(other, {"model": "gpt-5.6-sol"}, is_first_turn=True)
        assert other.model == "gpt-5.6-sol"
        assert other._cached_system_prompt == "Frozen synthetic prefix."
        assert not turn_routing.routing_enabled(agent.session_id)
    finally:
        scheduler._teardown_cron_agent(other, "other", timeout_seconds=0)


@pytest.mark.parametrize("inherited", [
    {}, {"model": None, "provider": None, "base_url": None},
    {"model": " ", "provider": "", "base_url": "\t"},
    {"model_snapshot": "gpt-6-astra", "provider_snapshot": "openai-codex"},
])
def test_inherited_defaults_are_not_job_pins(inherited: dict[str, Any], host_config: dict[str, Any]) -> None:
    agent = construct(inherited, host_config, "cron_public_default")
    try:
        assert turn_routing.apply_route(agent, {"model": "gpt-5.6-sol"}, is_first_turn=True)
        assert agent.model == "gpt-5.6-sol"
    finally:
        scheduler._teardown_cron_agent(agent, "public", timeout_seconds=0)


def test_constructor_failure_does_not_publish_pin(host_config: dict[str, Any]) -> None:
    def fail(**kwargs: Any) -> Any:
        raise RuntimeError("public construction failure")
    sid = "cron_public_failed"
    before = turn_routing.snapshot_routing_state(sid)
    with pytest.raises(RuntimeError, match="public construction failure"):
        scheduler._construct_cron_agent(
            fail, {"model": "gpt-6-astra"}, host_config,
            scheduler._CronAgentSetup(model="gpt-6-astra", runtime={"provider": "openai-codex"}),
            workdir=None, session_id=sid, session_db=None,
        )
    assert turn_routing.snapshot_routing_state(sid) == before


def test_job_pin_survives_failed_persistence_and_preserves_manual_effort(
    host_config: dict[str, Any], monkeypatch: Any,
) -> None:
    from agent.turn_reasoning import prepare_turn, set_session_reasoning
    sid = "cron_public_io_failure"
    assert set_session_reasoning(sid, {"enabled": True, "effort": "high"})
    monkeypatch.setattr(turn_routing, "atomic_json_write", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("public disk failure")))
    agent = construct({"provider": "openai-codex"}, host_config, sid)
    try:
        assert not turn_routing.apply_route(agent, {"model": "gpt-5.6-sol"}, is_first_turn=True)
        prepare_turn(agent)
        assert agent.reasoning_config == {"enabled": True, "effort": "high"}
        assert agent.reasoning_provenance == "session"
    finally:
        scheduler._teardown_cron_agent(agent, "public", timeout_seconds=0)
