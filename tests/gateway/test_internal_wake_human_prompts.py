"""Internal wake provenance must survive both gateway human-input guards."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.wake import deliver_wake
from tools import clarify_gateway as clarify


class Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.MATRIX)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


@pytest.fixture
def pipeline(monkeypatch):
    adapter = Adapter()
    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:test",
                           user_id="@requester:test", chat_type="dm")
    key = build_session_key(source)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._update_prompt_pending = {}
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized_for_source = lambda source: True
    runner._is_user_authorized = lambda source: True
    runner._session_key_for_source = lambda source: key
    runner._adapter_for_source = lambda source: adapter
    runner._hm_evict_idle_stale_agent = lambda key: None
    runner._hm_evict_reaped_agent = lambda key: None
    runner._is_session_running = lambda key: True
    # The real runner is used through interception/dispatch; replace only the
    # agent execution boundary, so no model, platform network, or live state runs.
    runner._hm_handle_running_session_message = AsyncMock(return_value=None)
    adapter._message_handler = runner._handle_message
    adapter._busy_session_handler = runner._handle_active_session_busy_message
    adapter._active_sessions[key] = asyncio.Event()
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *a, **kw: [])
    yield adapter, runner, source, key
    clarify.clear_session(key)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["wake", "runner"])
@pytest.mark.parametrize("choices", [None, ["A", "B"]])
@pytest.mark.parametrize("text", [
    "[ASYNC DELEGATION BATCH COMPLETE — deleg_dff90581]\nchild result",
    "[SYSTEM: Background process completed]\noutput",
    "[Cron job completed]\nscheduled result",
    "B",  # Provenance, not a prefix blacklist or a choice-value heuristic.
])
async def test_wakes_preserve_pending_clarify_and_human_answer(pipeline, route, choices, text):
    adapter, runner, source, key = pipeline
    entry = clarify.register("question", key, "Which?", choices)
    if route == "wake":
        await deliver_wake(adapter, source=source, text=text)
        queued = adapter._pending_messages[key]
        assert queued.internal and queued.text == text
        runner._hm_handle_running_session_message.assert_not_awaited()
    else:
        event = MessageEvent(text=text, source=source, internal=True)
        await runner._handle_message(event)
        runner._hm_handle_running_session_message.assert_awaited_once_with(event, source, key)
    assert not entry.event.is_set()
    assert entry.response is None
    # A requester can even type notification-looking text: it is still human input.
    answer = "2" if choices else text
    await adapter.handle_message(MessageEvent(text=answer, source=source))
    assert entry.response == ("B" if choices else text)
    assert clarify.wait_for_response("question", timeout=2) == entry.response


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["/approve", "!approve", "yes", "/deny"])
async def test_wake_cannot_dispatch_approval_but_human_can(pipeline, text):
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    adapter, runner, source, key = pipeline
    runner._hm_handle_running_session_message = (
        GatewayRunner._hm_handle_running_session_message.__get__(runner)
    )
    entry = _ApprovalEntry({"command": "test command", "request_id": "request"})
    approval._gateway_queues[key] = [entry]
    # Exercise the real approval resolver without platform confirmation UI.
    runner._deliver_approval_confirmation = AsyncMock(return_value="confirmed")
    runner._check_slash_access = lambda *args: None
    runner._effective_busy_input_mode = lambda source: "queue"
    runner._draining = False
    try:
        await deliver_wake(adapter, source=source, text=text)
        assert adapter._pending_messages[key].text == text
        assert approval.has_blocking_approval(key)
        assert not entry.event.is_set()
        # Also exercise plaintext routing directly, independent of adapter queuing.
        assert not await runner._route_plaintext_approval_while_busy(
            MessageEvent(text="yes", source=source, internal=True), key)
        assert not entry.event.is_set()
        await adapter.handle_message(MessageEvent(text="/approve", source=source))
        assert entry.event.is_set()
        assert entry.result == "once"
    finally:
        approval.unregister_gateway_notify(key)
