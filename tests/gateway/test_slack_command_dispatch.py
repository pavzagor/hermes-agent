"""Slack addressed commands must traverse the real adapter and runner without inference."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionEntry
from hermes_state import AsyncSessionDB
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def command_pipeline(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_HUMAN")
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="test", extra={"require_mention": False}))
    adapter._bot_user_id = "U_BOT"
    adapter._bot_display_name = "hermes"
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._app.client.users_info.return_value = {"user": {"profile": {"display_name": "Pavel"}}}
    adapter._app.client.conversations_info.return_value = {"channel": {"name": "test"}}
    adapter._app.client.conversations_replies.return_value = {"ok": True, "messages": []}
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="reply"))
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.SLACK: adapter.config})
    runner.adapters = {Platform.SLACK: adapter}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="test", session_id="slack-command-session", created_at=datetime.now(),
        updated_at=datetime.now(), platform=Platform.SLACK, chat_type="group")
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.get_session.return_value = None
    runner._session_db._db.get_session_title.return_value = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._handle_message_with_agent = AsyncMock(side_effect=AssertionError("command reached inference"))
    runner._claim_active_session_slot = MagicMock(side_effect=AssertionError("command started agent turn"))
    events = []
    results = []

    async def dispatch(event):
        events.append(event)
        result = await runner._handle_message(event)
        results.append(result)
        return result

    adapter.set_message_handler(dispatch)
    return adapter, runner, events, results


@pytest.mark.asyncio
@pytest.mark.parametrize("address", ["<@U_BOT>", "<@U_BOT|hermes>", "@hermes"])
@pytest.mark.parametrize("prefix", ["/", "!"])
@pytest.mark.parametrize("busy", [False, True])
async def test_addressed_status_dispatches_without_inference(command_pipeline, address, prefix, busy):
    adapter, runner, events, results = command_pipeline
    # Real Slack mentions must also pass the stricter channel gate. Literal
    # names normalize only in chats already admitted without a real mention.
    adapter.config.extra["require_mention"] = address.startswith("<@")
    source = adapter.build_source(chat_id="C_TEST", chat_type="group", user_id="U_HUMAN", thread_id="100.1")
    key = runner._session_key_for_source(source)
    if busy:
        adapter._active_sessions[key] = asyncio.Event()
        adapter._session_tasks[key] = asyncio.current_task()
        runner._running_agents[key] = SimpleNamespace(
            model="test-model", provider="test", interrupt=MagicMock(),
            context_compressor=SimpleNamespace(last_prompt_tokens=0, context_length=1000))
    await adapter._handle_slack_message({
        "type": "message", "channel": "C_TEST", "channel_type": "channel",
        "user": "U_HUMAN", "text": f"{address} {prefix}status", "ts": "200.1",
        "thread_ts": "100.1",
    })
    if adapter._background_tasks:
        await asyncio.gather(*adapter._background_tasks)
    assert len(events) == 1
    assert events[0].text == "/status"
    assert events[0].source.thread_id == "100.1"
    assert results and "slack-command-session" in results[0]
    runner._claim_active_session_slot.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()
    assert adapter.send.await_count == 1
    assert adapter.send.call_args.kwargs["metadata"]["thread_id"] == "100.1"
    assert runner._session_key_for_source(events[0].source) == key
    if busy:
        assert key in adapter._active_sessions
        assert key not in adapter._pending_messages
        runner._running_agents[key].interrupt.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("case,text", [
    ("other-user", "<@U_OTHER|hermes> /status"),
    ("name-prefix", "@hermes-other /status"),
    ("prose", "@hermes explain /status"),
    ("unknown-bang", "@hermes !nice work"),
    ("not-leading", "ask @hermes /status"),
    ("unauthorized", "@hermes /status"),
    ("mention-required", "@hermes /status"),
    ("internal", "<@U_BOT|hermes> /status"),
    ("no-control", "<@U_BOT|hermes> /status"),
])
async def test_command_normalization_preserves_admission_and_control_guards(command_pipeline, case, text):
    adapter, runner, events, results = command_pipeline
    runner._claim_active_session_slot.side_effect = None
    runner._claim_active_session_slot.return_value = (None, "ordinary chat")
    adapter.config.extra["require_mention"] = case == "mention-required"
    original_dispatch = adapter._message_handler

    async def dispatch(event):
        event.internal = case == "internal"
        event.allow_gateway_control = case != "no-control"
        return await original_dispatch(event)

    adapter.set_message_handler(dispatch)
    await adapter._handle_slack_message({
        "type": "message", "channel": "C_TEST", "channel_type": "channel",
        "user": "U_DENIED" if case == "unauthorized" else "U_HUMAN",
        "text": text, "ts": "200.1", "thread_ts": "100.1",
    })
    if adapter._background_tasks:
        await asyncio.gather(*adapter._background_tasks)
    if case == "mention-required":
        assert not events
        assert not results
        adapter.send.assert_not_awaited()
    elif case == "unauthorized":
        assert len(events) == 1
        assert results == [None]
        runner._claim_active_session_slot.assert_not_called()
        adapter.send.assert_not_awaited()
    else:
        assert len(events) == 1
        assert events[0].get_command() is None
        assert results == ["ordinary chat"]
    runner._handle_message_with_agent.assert_not_awaited()
