"""Gateway plugin commands receive the persistent conversation identity."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource


@pytest.mark.asyncio
async def test_gateway_plugin_command_binds_persistent_session(monkeypatch):
    from gateway.run import GatewayRunner
    from hermes_cli import plugins

    source = SessionSource(
        platform=Platform.TELEGRAM, user_id="u", chat_id="c", chat_type="dm")
    entry = SessionEntry(
        session_key="telegram:c", session_id="db-session",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm")
    runner = object.__new__(GatewayRunner)
    runner._draining = False
    from gateway.config import GatewayConfig
    runner.config = GatewayConfig()
    runner._hm_quick_commands = lambda: {}
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=entry))
    seen = {}
    def handler(raw_args):
        seen["args"] = raw_args
        seen["context"] = plugins.current_plugin_command_context()
        return "ok"
    monkeypatch.setattr(plugins, "get_plugin_command_handler", lambda name: handler)
    event = SimpleNamespace(get_command_args=lambda: "on")

    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(
        event, source, "routing", "telegram:c")

    assert (handled, result, command) == (True, "ok", "routing")
    assert (seen["args"], seen["context"].session_id, seen["context"].session_key) == (
        "on", "db-session", "telegram:c")
    assert plugins.current_plugin_command_context() is None
