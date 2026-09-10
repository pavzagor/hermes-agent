"""Reaction intake through the real Matrix text pipeline, with a fake homeserver."""
import asyncio
import re
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from plugins.platforms.matrix.adapter import MatrixAdapter

ROOM = "!room:example.org"
BOT = "@bot:example.org"
USER = "@alice:example.org"


def adapter_for(monkeypatch, enabled=True, dm=False):
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", USER)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    from gateway.config import GatewayConfig, Platform
    from gateway.config_loader import merge_platform_sections

    yaml_config = {"gateway": {"platforms": {"matrix": {"enabled": True, "extra": {
        "homeserver": "https://example.org", "user_id": BOT,
        **({"forward_reactions": enabled} if enabled is not None else {}),
    }}}}}
    gateway_data = {}
    merge_platform_sections(yaml_config, yaml_config["gateway"], gateway_data)
    adapter = MatrixAdapter(GatewayConfig.from_dict(gateway_data).platforms[Platform.MATRIX])
    adapter._startup_ts = time.time()
    adapter._text_batch_delay_seconds = 0
    adapter._resolve_room_identity = AsyncMock(return_value=SimpleNamespace(
        display_name="Room", room_topic="Topic", server_name="example.org",
        chat_type="dm" if dm else "group"))
    adapter._get_display_name = AsyncMock(side_effect=lambda room, sender: sender)
    adapter._background_read_receipt = Mock()
    adapter.handle_message = AsyncMock()
    return adapter


def reaction(target="$target", sender=USER, event_id="$reaction", key="👍"):
    return SimpleNamespace(room_id=ROOM, sender=sender, event_id=event_id,
        timestamp=int(time.time() * 1000), content={"m.relates_to": {
            "rel_type": "m.annotation", "event_id": target, "key": key}})


@pytest.mark.asyncio
@pytest.mark.parametrize("dm,thread,auto", [
    (False, "$root", True), (False, None, False), (False, None, True),
    (True, "$root", False), (True, None, False), (True, None, True),
])
@pytest.mark.parametrize("batch", [False, True])
async def test_authorized_reaction_fetches_target_and_reuses_text_intake(monkeypatch, dm, thread, auto, batch):
    adapter = adapter_for(monkeypatch, dm=dm)
    adapter._require_mention = True
    adapter._thread_require_mention = True
    adapter._auto_thread = auto
    adapter._dm_auto_thread = auto
    # A reaction is directed feedback, not an actual @mention that starts a DM thread.
    adapter._dm_mention_threads = True
    if batch:
        adapter._text_batch_delay_seconds = 0.001
    target = SimpleNamespace(sender=BOT, type="m.room.message", room_id=ROOM,
        content={"msgtype": "m.text", "body": "Here is the answer.",
                 "m.relates_to": {"rel_type": "m.thread", "event_id": thread} if thread else {}})
    adapter._client = SimpleNamespace(get_event=AsyncMock(return_value=target))
    incoming = reaction()
    await adapter._on_reaction(incoming)
    await adapter._on_reaction(incoming)
    if batch:
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())
    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.source.chat_id == ROOM
    assert msg.source.thread_id == (thread or ("$target" if auto else None))
    assert msg.source.user_id == msg.user_id == USER
    assert msg.message_id == "$reaction"
    assert msg.reply_to_message_id == "$target"
    assert msg.reply_to_text == target.content["body"]
    assert msg.reply_to_author_id == BOT
    assert msg.allow_gateway_control is False
    assert "👍" in msg.text and "not authorization" in msg.text
    adapter._client.get_event.assert_any_await(ROOM, "$target")
    if batch:
        from dataclasses import replace
        adapter.handle_message.reset_mock()
        typed = replace(msg, text="A typed answer", allow_gateway_control=True)
        # The shared batcher must not lend typed-answer authority to feedback.
        adapter._enqueue_text_event(typed)
        adapter._enqueue_text_event(msg)
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())
        assert [(c.args[0].text, c.args[0].allow_gateway_control)
                for c in adapter.handle_message.await_args_list] == [
                    ("A typed answer", True), (msg.text, False)]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", [
    "default", "disabled", "string-false", "unauthorized", "self", "bridge", "ignored", "room",
    "old", "malformed", "other-author", "fetch-error", "redacted", "encrypted",
    "legacy-approval", "legacy-model", "legacy-choice", "legacy-clarify",
    "retired-approval", "retired-model", "retired-choice", "retired-clarify",
])
async def test_reaction_intake_fails_closed(monkeypatch, blocked):
    adapter = adapter_for(monkeypatch, enabled={"default": None, "disabled": False, "string-false": "false"}.get(blocked, True))
    adapter._auto_thread = False
    content = {"msgtype": "m.text", "body": "An answer"}
    target = SimpleNamespace(sender=BOT, type="m.room.message", room_id=ROOM, content=content)
    event = reaction()
    if blocked in {"unauthorized", "self", "bridge", "ignored"}:
        event.sender = {"unauthorized": "@mallory:example.org", "self": BOT,
                        "bridge": "@_bridge:example.org", "ignored": USER}[blocked]
        if blocked == "bridge":
            adapter._allowed_user_ids.add(event.sender)
        if blocked == "ignored":
            adapter._ignored_user_patterns = [re.compile("alice")]
    if blocked == "room":
        adapter._allowed_room_ids = {"!other:example.org"}
    if blocked == "old":
        event.timestamp = 1000
    if blocked == "malformed":
        event.content["m.relates_to"]["rel_type"] = "m.replace"
    if blocked == "other-author":
        target.sender = USER
    if blocked == "redacted":
        target.content = {}
    if blocked == "encrypted":
        target.type = "m.room.encrypted"
    legacy = {"legacy-approval": "You can also click the reaction to approve:",
              "legacy-model": "React to choose a model:", "legacy-choice": "React to choose.",
              "legacy-clarify": "❓ Proceed?\nChoose an option, or type an answer:"}
    if blocked in legacy:
        content["body"] = legacy[blocked]
    if blocked.startswith("retired-"):
        # No volatile prompt registry: this is a target fetched after restart.
        content["org.hermes.control_prompt"] = blocked.removeprefix("retired-")
    adapter._client = SimpleNamespace(get_event=AsyncMock(return_value=target))
    if blocked == "fetch-error":
        adapter._client.get_event.side_effect = RuntimeError("unavailable")
    await adapter._on_reaction(event)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["approval", "model", "choice", "clarify"])
async def test_real_control_transport_keeps_retired_prompts_out_of_intake(monkeypatch, kind):
    adapter = adapter_for(monkeypatch)
    adapter._schedule_reaction_redaction = Mock()
    posted = {}

    async def send_event(room, event_type, content):
        event_id = f"$sent-{len(posted)}"
        posted[event_id] = SimpleNamespace(sender=BOT, room_id=str(room),
            type=str(event_type), content=content)
        return event_id

    adapter._client = SimpleNamespace(send_message_event=send_event,
        get_event=AsyncMock(side_effect=lambda room, eid: posted[eid]))
    metadata = {"thread_id": "$root", "requester_user_id": USER}
    if kind == "approval":
        result = await adapter.send_exec_approval(ROOM, "test-command", "session", metadata=metadata)
    elif kind == "model":
        result = await adapter.send_model_picker(ROOM,
            [{"slug": "test", "name": "Test", "models": ["test-model"]}],
            "test-model", "test", "session", AsyncMock(), metadata=metadata)
    elif kind == "choice":
        result = await adapter.send_choice_picker(ROOM, "Pick",
            [{"value": "yes", "label": "Yes"}], "session", AsyncMock(), metadata=metadata)
    else:
        result = await adapter.send_clarify(ROOM, "Pick", ["Yes"], "clarify", "session", metadata=metadata)
    assert result.success
    target = posted[result.message_id]
    assert target.content["org.hermes.control_prompt"]
    assert target.content["m.relates_to"]["event_id"] == "$root"
    assert "_matrix_control_prompt" not in metadata
    adapter._send_invalid_reaction_feedback = AsyncMock()
    # Invalid keys on active controls must be consumed BEFORE ordinary forwarding.
    await adapter._on_reaction(reaction(result.message_id, key="not-a-choice"))
    adapter._client.get_event.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()
    # Expiry consumes the first click and removes the volatile registry entry.
    registry = getattr(adapter, {
        "approval": "_approval_prompts_by_event", "model": "_model_picker_prompts_by_event",
        "choice": "_choice_picker_prompts_by_event", "clarify": "_clarify_prompts_by_event",
    }[kind])
    registry[result.message_id].expires_at = time.monotonic() - 1
    await adapter._on_reaction(reaction(result.message_id, event_id="$expire"))
    assert result.message_id not in registry
    # A new adapter represents restart; target safety comes from homeserver content.
    restarted = adapter_for(monkeypatch)
    restarted._client = adapter._client
    await restarted._on_reaction(reaction(result.message_id, event_id="$late"))
    restarted._client.get_event.assert_awaited_once_with(ROOM, result.message_id)
    restarted.handle_message.assert_not_awaited()
