"""Real consumer/Slack adapter regression for formatted, media-accompanied finals.

Related upstream fix: https://github.com/NousResearch/hermes-agent/pull/98231
Only Slack I/O is fake; no live Slack credentials or API calls.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,fail_edits", [("draft", False), ("edit", False), ("edit", True)])
async def test_formatted_media_final_replaces_partial(transport, fail_edits):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra={}))
    adapter._app = MagicMock()
    adapter.stop_typing = AsyncMock()
    client = AsyncMock()
    adapter._get_client = MagicMock(return_value=client)
    messages = {}
    started = asyncio.Event()

    async def post(**kwargs):
        ts = str(len(messages) + 1)
        messages[ts] = kwargs.get("text", kwargs.get("markdown_text", ""))
        started.set()
        return {"ok": True, "ts": ts}

    async def edit(**kwargs):
        if fail_edits:
            raise TimeoutError("simulated chat.update transport failure")
        messages[kwargs["ts"]] = kwargs["text"]
        return {"ok": True}

    async def append(**kwargs):
        messages[kwargs["ts"]] += kwargs.get("markdown_text", "")
        return {"ok": True}

    async def delete(**kwargs):
        messages.pop(kwargs["ts"], None)
        return {"ok": True}

    client.chat_postMessage.side_effect = post
    client.chat_startStream.side_effect = post
    client.chat_update.side_effect = edit
    client.chat_appendStream.side_effect = append
    client.chat_stopStream.side_effect = append
    client.chat_delete.side_effect = delete
    consumer = GatewayStreamConsumer(
        adapter, "D1", metadata={"thread_id": "111.000", "user_id": "U123"},
        config=StreamConsumerConfig(transport=transport, edit_interval=60, buffer_threshold=10000),
    )
    text = "**Yes—and ElevenLabs already returns MP3.** I checked our adapter: it requests `mp3_44100_128`."
    partial = text[:text.index("00_128")]
    final = text + "\n\nMEDIA:/tmp/speech.ogg [[audio_as_voice]]"
    task = asyncio.create_task(consumer.run())
    try:
        consumer.on_delta(partial)
        await asyncio.wait_for(started.wait(), timeout=5)
        consumer.on_delta(final[len(partial):])
        consumer.finish(final)
        await asyncio.wait_for(task, timeout=5)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert consumer.final_content_delivered
    assert consumer.delivered_final_matches(final)
    assert len(messages) == 1, messages
    assert all(consumer.cfg.cursor not in value for value in messages.values())
    assert all("MEDIA:" not in value for value in messages.values())
