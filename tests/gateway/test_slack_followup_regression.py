"""PR #100992 follow-up: local-only frame matrix and retry characterization."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from agent.stream_delivery import StreamDeliveryMixin
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.test_slack_native_streaming import META, _make_adapter


@pytest.mark.asyncio
@pytest.mark.parametrize('leading', ['', '\n\n'])
@pytest.mark.parametrize('frame,final', [
    ('abc', 'abc — more'),
    ('abc\n', 'abc — more'),
    ('abc\n', 'abc'),
    ('abc \t\n', 'abc'),
    ('removed `.env.bak ▉`', 'removed `.env.bak-2026` and done'),
    ('```\nls ▉\n```', '```\nls -la\n```'),
])
async def test_frame_final_matrix(leading, frame, final):
    adapter, client = _make_adapter()
    await adapter.send_draft('D1', 7, leading + frame, metadata=META)
    result = await adapter.send('D1', final, metadata=META)
    assert result.success
    assert client.chat_startStream.await_count == 1
    assert client.chat_postMessage.await_count == 0
    assert client.chat_stopStream.await_count == 1
    assert not adapter._active_streams


@pytest.mark.asyncio
@pytest.mark.parametrize('after_tool', [False, True])
@pytest.mark.parametrize('frame,final', [
    ('Partial answer already streamed', 'Partial answer already streamed — more'),
    ('Complete answer already streamed\n', 'Complete answer already streamed'),
    ('removed `.env.bak', 'removed `.env.bak-2026` and done'),
    ('```\nls', '```\nls -la\n```'),
])
async def test_real_consumer_partial_final(after_tool, frame, final):
    adapter, client = _make_adapter()
    started = asyncio.Event()
    async def start(**kwargs):
        started.set()
        return {'ok': True, 'ts': '123.456'}
    client.chat_startStream.side_effect = start
    consumer = GatewayStreamConsumer(adapter, 'D1', metadata=META,
        config=StreamConsumerConfig(transport='draft', edit_interval=0.01, buffer_threshold=1))
    agent = StreamDeliveryMixin()
    agent.stream_delta_callback = consumer.on_delta
    agent._stream_callback = None
    agent._strip_think_blocks = lambda text: text
    agent._stream_needs_break = after_tool
    task = asyncio.create_task(consumer.run())
    try:
        agent._fire_stream_delta(frame)
        await asyncio.wait_for(started.wait(), 3)
        consumer.finish(final)
        await asyncio.wait_for(task, 3)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert client.chat_startStream.await_count == 1
    assert client.chat_postMessage.await_count == 0
    assert client.chat_stopStream.await_count == 1
    assert consumer.delivered_final_matches(final)
    assert not adapter._active_streams


@pytest.mark.asyncio
async def test_identical_draft_retry_is_noop():
    adapter, client = _make_adapter()
    for _ in range(2):
        assert (await adapter.send_draft('D1', 7, 'abc\n', metadata=META)).success
    assert client.chat_startStream.await_count == 1
    assert client.chat_appendStream.await_count == 0


@pytest.mark.asyncio
async def test_final_retry_is_not_idempotent_characterization():
    # This is a known remaining limitation, NOT a no-duplicate guarantee.
    adapter, client = _make_adapter()
    await adapter.send_draft('D1', 7, 'abc', metadata=META)
    await adapter.send('D1', 'abc', metadata=META)
    await adapter.send('D1', 'abc', metadata=META)
    assert client.chat_stopStream.await_count == 1
    assert client.chat_postMessage.await_count == 1


@pytest.mark.asyncio
async def test_stop_timeout_fallback_can_duplicate_characterization():
    # An ambiguous timeout after server acceptance cannot be inferred locally.
    adapter, client = _make_adapter()
    await adapter.send_draft('D1', 7, 'abc', metadata=META)
    client.chat_stopStream = AsyncMock(side_effect=TimeoutError('ambiguous acceptance'))
    assert (await adapter.send('D1', 'abc', metadata=META)).success
    assert client.chat_stopStream.await_count == 1
    assert client.chat_postMessage.await_count == 1


@pytest.mark.asyncio
async def test_append_timeout_retry_starts_another_stream_characterization():
    adapter, client = _make_adapter()
    await adapter.send_draft('D1', 7, 'abc', metadata=META)
    client.chat_appendStream.side_effect = TimeoutError('ambiguous acceptance')
    assert not (await adapter.send_draft('D1', 7, 'abcdef', metadata=META)).success
    assert (await adapter.send_draft('D1', 7, 'abcdef', metadata=META)).success
    assert client.chat_startStream.await_count == 2
    assert client.chat_stopStream.await_count == 0
