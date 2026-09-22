"""Real consumer/adapter regression; only Slack IO is faked."""
import asyncio
import pytest
from agent.stream_delivery import StreamDeliveryMixin
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.test_slack_native_streaming import META, _make_adapter

FINAL = ('**Neither account is created yet.**\n\n'
         '- **WorkOS:** browser automation repeatedly disconnected while opening signup, before I could submit your email. No confirmation email was triggered by me.\n'
         '- **“coerk”:** did you mean **Clerk — clerk.com**? Please confirm the service so I don’t register you on the wrong site.\n')

@pytest.mark.asyncio
@pytest.mark.parametrize('after_tool', [False, True])
@pytest.mark.parametrize('complete', [False, True])
async def test_final_exactly_once(after_tool, complete):
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
        agent._fire_stream_delta(FINAL if complete else FINAL[:80])
        await asyncio.wait_for(started.wait(), 3)
        consumer.finish(FINAL)
        await asyncio.wait_for(task, 3)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    wire = [(c[0], c.kwargs) for c in client.mock_calls]
    assert client.chat_postMessage.await_count == 0, wire
    assert client.chat_startStream.await_count == 1, wire
    assert client.chat_stopStream.await_count == 1, wire
    markdown = ''.join(c.kwargs.get('markdown_text', '') for c in client.mock_calls
                       if c[0] in ('chat_startStream', 'chat_appendStream', 'chat_stopStream'))
    assert markdown.strip() == FINAL.strip(), wire
    assert consumer.delivered_final_matches(FINAL)
