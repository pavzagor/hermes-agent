"""Slack public-channel auto-join behavior."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.mark.asyncio
async def test_channel_created_joins_public_channel_in_event_workspace():
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    client = SimpleNamespace(conversations_join=AsyncMock())
    adapter._team_clients = {"T123": client}

    await adapter._handle_channel_created(
        {"channel": {"id": "C123", "is_private": False}},
        {"team_id": "T123"},
    )

    client.conversations_join.assert_awaited_once_with(channel="C123")


@pytest.mark.asyncio
async def test_channel_created_does_not_join_private_channel():
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    client = SimpleNamespace(conversations_join=AsyncMock())
    adapter._team_clients = {"T123": client}

    await adapter._handle_channel_created(
        {"channel": {"id": "G123", "is_private": True}},
        {"team_id": "T123"},
    )

    client.conversations_join.assert_not_awaited()
