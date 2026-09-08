"""Focused tests for Matrix reaction-based generic clarify prompts."""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.matrix.adapter import MatrixAdapter
from tools import clarify_gateway


ROOM_ID = "!clarify:example.org"
OTHER = "✍️"
REQUESTER = "@alice:example.org"


def _clear_clarify_gateway_state() -> None:
    with clarify_gateway._lock:
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()
        clarify_gateway._notify_cbs.clear()


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> MatrixAdapter:
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", REQUESTER)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"homeserver": "https://matrix.example.org"},
        )
    )
    adapter._client = SimpleNamespace()
    adapter._user_id = "@bot:example.org"
    adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="$clarify-prompt")
    )
    adapter._send_reaction = AsyncMock(
        side_effect=lambda _room, _event, emoji: f"$seed-{emoji}"
    )
    return adapter


def _reaction_event(
    key: str,
    *,
    sender: str = REQUESTER,
    room_id: str = ROOM_ID,
    event_id: str = "$user-reaction",
) -> SimpleNamespace:
    return SimpleNamespace(
        sender=sender,
        event_id=event_id,
        room_id=room_id,
        content={
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": "$clarify-prompt",
                "key": key,
            }
        },
    )


@pytest.fixture(autouse=True)
def _reset_clarify_state():
    _clear_clarify_gateway_state()
    yield
    _clear_clarify_gateway_state()


@pytest.mark.asyncio
async def test_leading_choice_emoji_seed_real_reactions_plus_other(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    choices = ["🚀 Execute", "👀 Pause", "✋ Stop", "👍 Proceed"]

    result = await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="What next?",
        choices=choices,
        clarify_id="clarify-leading",
        session_key="matrix:leading",
        metadata={"requester_user_id": REQUESTER},
    )

    assert result.success is True
    adapter.send.assert_awaited_once()
    send_call = adapter.send.await_args
    prompt_body = (
        send_call.args[1] if len(send_call.args) > 1 else send_call.kwargs["content"]
    )
    assert all(choice in prompt_body for choice in choices)
    assert [call.args[2] for call in adapter._send_reaction.await_args_list] == [
        "🚀",
        "👀",
        "✋",
        "👍",
        OTHER,
    ]
    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]
    assert adapter._clarify_prompt_event_by_id["clarify-leading"] == "$clarify-prompt"
    assert prompt.chat_id == ROOM_ID
    assert prompt.clarify_id == "clarify-leading"
    assert prompt.session_key == "matrix:leading"
    assert prompt.choices == dict(zip(["🚀", "👀", "✋", "👍"], choices))
    assert prompt.requester_user_id == REQUESTER
    assert prompt.resolved is False
    assert prompt.bot_reaction_events[OTHER] == f"$seed-{OTHER}"


@pytest.mark.asyncio
async def test_choices_without_emoji_seed_numeric_reactions(monkeypatch):
    adapter = _make_adapter(monkeypatch)

    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick one",
        choices=["Alpha", "Beta", "Gamma", "Delta"],
        clarify_id="clarify-numeric",
        session_key="matrix:numeric",
    )

    assert [call.args[2] for call in adapter._send_reaction.await_args_list] == [
        "1️⃣",
        "2️⃣",
        "3️⃣",
        "4️⃣",
        OTHER,
    ]


@pytest.mark.asyncio
async def test_duplicate_and_reserved_leading_emoji_fall_back_only_for_conflicts(
    monkeypatch,
):
    adapter = _make_adapter(monkeypatch)
    choices = ["🚀 Execute", "🚀 Execute later", "✍️ Type it", "👍 Proceed"]

    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick one",
        choices=choices,
        clarify_id="clarify-conflicts",
        session_key="matrix:conflicts",
    )

    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]
    assert prompt.choices == {
        "🚀": "🚀 Execute",
        "2️⃣": "🚀 Execute later",
        "3️⃣": "✍️ Type it",
        "👍": "👍 Proceed",
    }
    assert [call.args[2] for call in adapter._send_reaction.await_args_list] == [
        "🚀",
        "2️⃣",
        "3️⃣",
        "👍",
        OTHER,
    ]


@pytest.mark.asyncio
async def test_reserved_other_presentation_variants_use_numeric_fallback(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    choices = ["✍ Type it", "✍︎ Type text", "✍️ Type emoji"]

    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick one",
        choices=choices,
        clarify_id="clarify-reserved-variants",
        session_key="matrix:reserved-variants",
    )

    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]
    assert prompt.choices == dict(zip(["1️⃣", "2️⃣", "3️⃣"], choices))
    assert [call.args[2] for call in adapter._send_reaction.await_args_list] == [
        "1️⃣",
        "2️⃣",
        "3️⃣",
        OTHER,
    ]


@pytest.mark.asyncio
async def test_duplicate_emoji_presentation_variants_use_numeric_fallback(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    choices = ["✋ Stop", "✋︎ Pause", "✋️ Wait"]

    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick one",
        choices=choices,
        clarify_id="clarify-duplicate-variants",
        session_key="matrix:duplicate-variants",
    )

    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]
    assert prompt.choices == {
        "✋": choices[0],
        "2️⃣": choices[1],
        "3️⃣": choices[2],
    }
    assert [call.args[2] for call in adapter._send_reaction.await_args_list] == [
        "✋",
        "2️⃣",
        "3️⃣",
        OTHER,
    ]


@pytest.mark.asyncio
async def test_valid_reaction_resolves_exact_canonical_choice(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    canonical_choice = "🚀 Execute without changing this text"
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Proceed?",
        choices=[canonical_choice, "Pause"],
        clarify_id="clarify-valid",
        session_key="matrix:valid",
        metadata={"requester_user_id": REQUESTER},
    )
    adapter._redact_bot_clarify_reactions = AsyncMock()

    with patch(
        "tools.clarify_gateway.resolve_gateway_clarify", return_value=True
    ) as resolve:
        await adapter._on_reaction(_reaction_event("🚀"))

    resolve.assert_called_once_with("clarify-valid", canonical_choice)
    assert "$clarify-prompt" not in adapter._clarify_prompts_by_event
    adapter._redact_bot_clarify_reactions.assert_awaited_once()


@pytest.mark.asyncio
async def test_choice_reaction_presentation_variant_resolves_canonical_choice(
    monkeypatch,
):
    adapter = _make_adapter(monkeypatch)
    canonical_choice = "✋ Stop with canonical text"
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Proceed?",
        choices=[canonical_choice, "Pause"],
        clarify_id="clarify-choice-variant",
        session_key="matrix:choice-variant",
        metadata={"requester_user_id": REQUESTER},
    )
    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]
    assert prompt.choices["✋"] == canonical_choice
    adapter._redact_bot_clarify_reactions = AsyncMock()

    with patch(
        "tools.clarify_gateway.resolve_gateway_clarify", return_value=True
    ) as resolve:
        await adapter._on_reaction(_reaction_event("✋️"))

    resolve.assert_called_once_with("clarify-choice-variant", canonical_choice)
    assert "$clarify-prompt" not in adapter._clarify_prompts_by_event
    adapter._redact_bot_clarify_reactions.assert_awaited_once_with(ROOM_ID, prompt)


@pytest.mark.asyncio
async def test_clarify_cleanup_delays_only_bot_seed_redactions(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    adapter._reaction_redaction_delay_seconds = 0.01
    adapter._redact_reaction = AsyncMock(return_value=True)
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick",
        choices=["Alpha", "Beta"],
        clarify_id="clarify-cleanup",
        session_key="matrix:cleanup",
    )
    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]

    await adapter._redact_bot_clarify_reactions(ROOM_ID, prompt)

    adapter._redact_reaction.assert_not_awaited()
    await asyncio.sleep(0.03)
    redacted_event_ids = {
        call.args[1] for call in adapter._redact_reaction.await_args_list
    }
    assert redacted_event_ids == set(prompt.bot_reaction_events.values())
    assert "$user-reaction" not in redacted_event_ids
    assert all(
        call.args[2] == "clarify resolved"
        for call in adapter._redact_reaction.await_args_list
    )


@pytest.mark.asyncio
async def test_cleanup_clarify_retires_typed_prompt_and_schedules_seed_cleanup(
    monkeypatch,
):
    adapter = _make_adapter(monkeypatch)
    adapter._schedule_reaction_redaction = Mock()
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick",
        choices=["Alpha", "Beta"],
        clarify_id="clarify-typed-cleanup",
        session_key="matrix:typed-cleanup",
    )
    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]

    await adapter.cleanup_clarify("clarify-typed-cleanup")

    assert prompt.resolved is True
    assert adapter._clarify_prompts_by_event == {}
    assert adapter._clarify_prompt_event_by_id == {}
    assert {
        call.args[1] for call in adapter._schedule_reaction_redaction.call_args_list
    } == set(prompt.bot_reaction_events.values())


@pytest.mark.asyncio
async def test_cleanup_clarify_is_idempotent(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    adapter._schedule_reaction_redaction = Mock()
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick",
        choices=["Alpha"],
        clarify_id="clarify-idempotent",
        session_key="matrix:idempotent",
    )
    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]

    await adapter.cleanup_clarify("clarify-idempotent")
    first_cleanup_count = adapter._schedule_reaction_redaction.call_count
    await adapter.cleanup_clarify("clarify-idempotent")

    assert first_cleanup_count == len(prompt.bot_reaction_events)
    assert adapter._schedule_reaction_redaction.call_count == first_cleanup_count


@pytest.mark.asyncio
async def test_reaction_resolution_then_gateway_cleanup_does_not_double_redact(
    monkeypatch,
):
    adapter = _make_adapter(monkeypatch)
    adapter._schedule_reaction_redaction = Mock()
    clarify_gateway.register(
        "clarify-reaction-cleanup",
        "matrix:reaction-cleanup",
        "Pick",
        ["Alpha"],
    )
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick",
        choices=["Alpha"],
        clarify_id="clarify-reaction-cleanup",
        session_key="matrix:reaction-cleanup",
    )
    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]

    await adapter._on_reaction(_reaction_event("1️⃣"))
    await adapter.cleanup_clarify("clarify-reaction-cleanup")

    assert adapter._clarify_prompts_by_event == {}
    assert adapter._clarify_prompt_event_by_id == {}
    assert adapter._schedule_reaction_redaction.call_count == len(
        prompt.bot_reaction_events
    )


@pytest.mark.asyncio
async def test_disconnect_clears_pending_clarify_indexes_without_network_cleanup(
    monkeypatch,
):
    adapter = _make_adapter(monkeypatch)
    adapter._redact_reaction = AsyncMock(return_value=True)
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick",
        choices=["Alpha"],
        clarify_id="clarify-disconnect",
        session_key="matrix:disconnect",
    )
    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]
    adapter._client = None

    await adapter.disconnect()

    assert prompt.resolved is True
    assert adapter._clarify_prompts_by_event == {}
    assert adapter._clarify_prompt_event_by_id == {}
    adapter._redact_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_clarify_question_choice_and_response_are_not_logged(monkeypatch, caplog):
    adapter = _make_adapter(monkeypatch)
    question = "private-question-marker"
    choice = "🚀 private-choice-and-response-marker"
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question=question,
        choices=[choice],
        clarify_id="clarify-log-privacy",
        session_key="matrix:log-privacy",
        metadata={"requester_user_id": REQUESTER},
    )
    adapter._redact_bot_clarify_reactions = AsyncMock()

    with (
        caplog.at_level(logging.DEBUG),
        patch(
            "tools.clarify_gateway.resolve_gateway_clarify",
            return_value=True,
        ),
    ):
        await adapter._on_reaction(_reaction_event("🚀"))

    assert question not in caplog.text
    assert choice not in caplog.text
    assert "private-choice-and-response-marker" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "incoming_other",
    ["✍", "✍︎", "✍️"],
    ids=["bare", "text-presentation", "emoji-presentation"],
)
async def test_other_presentation_variants_enter_text_mode_without_resolving(
    monkeypatch,
    incoming_other,
):
    adapter = _make_adapter(monkeypatch)
    clarify_gateway.register("clarify-other", "matrix:other", "Pick", ["Alpha", "Beta"])
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick",
        choices=["Alpha", "Beta"],
        clarify_id="clarify-other",
        session_key="matrix:other",
        metadata={"requester_user_id": REQUESTER},
    )
    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]
    assert set(prompt.bot_reaction_events) == {"1️⃣", "2️⃣", OTHER}
    adapter._redact_bot_clarify_reactions = AsyncMock()

    with patch("tools.clarify_gateway.resolve_gateway_clarify") as resolve:
        await adapter._on_reaction(_reaction_event(incoming_other))

    resolve.assert_not_called()
    entry = clarify_gateway._entries["clarify-other"]
    assert entry.awaiting_text is True
    assert entry.response is None
    assert entry.event.is_set() is False
    assert "$clarify-prompt" not in adapter._clarify_prompts_by_event
    adapter._redact_bot_clarify_reactions.assert_awaited_once_with(ROOM_ID, prompt)


@pytest.mark.asyncio
async def test_open_ended_prompt_seeds_no_reactions(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    clarify_gateway.register("clarify-open", "matrix:open", "Explain", None)

    result = await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Explain your preference",
        choices=None,
        clarify_id="clarify-open",
        session_key="matrix:open",
    )

    assert result.success is True
    adapter._send_reaction.assert_not_awaited()
    assert adapter._clarify_prompts_by_event == {}
    assert clarify_gateway._entries["clarify-open"].awaiting_text is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "mutate_prompt"),
    [
        (_reaction_event("1️⃣", sender="@mallory:example.org"), None),
        (_reaction_event("1️⃣", room_id="!wrong:example.org"), None),
        (_reaction_event("🛑"), None),
        (_reaction_event("1️⃣"), "expire"),
    ],
    ids=["unauthorized", "wrong-room", "invalid-choice", "expired"],
)
async def test_invalid_clarify_reactions_fail_closed(monkeypatch, event, mutate_prompt):
    adapter = _make_adapter(monkeypatch)
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick",
        choices=["Alpha", "Beta"],
        clarify_id="clarify-invalid",
        session_key="matrix:invalid",
        metadata={"requester_user_id": REQUESTER},
    )
    adapter._redact_bot_clarify_reactions = AsyncMock()
    adapter._send_invalid_reaction_feedback = AsyncMock()
    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]
    if mutate_prompt == "expire":
        prompt.expires_at = time.monotonic() - 1

    with patch(
        "tools.clarify_gateway.resolve_gateway_clarify", return_value=True
    ) as resolve:
        await adapter._on_reaction(event)

    resolve.assert_not_called()
    if mutate_prompt == "expire":
        assert prompt.resolved is True
        assert "$clarify-prompt" not in adapter._clarify_prompts_by_event
        adapter._redact_bot_clarify_reactions.assert_awaited_once_with(ROOM_ID, prompt)
        adapter._send_invalid_reaction_feedback.assert_awaited_once_with(
            ROOM_ID,
            "$clarify-prompt",
            "This clarify prompt has expired. Ask again to choose an answer.",
        )
    else:
        assert prompt.resolved is False
        assert "$clarify-prompt" in adapter._clarify_prompts_by_event


@pytest.mark.asyncio
async def test_globally_allowed_non_requester_cannot_resolve_clarify(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    bob = "@bob:example.org"
    adapter._allowed_user_ids = {REQUESTER, bob}
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick",
        choices=["Alpha"],
        clarify_id="clarify-requester-only",
        session_key="matrix:requester-only",
        metadata={"requester_user_id": REQUESTER},
    )
    adapter._send_invalid_reaction_feedback = AsyncMock()

    with patch("tools.clarify_gateway.resolve_gateway_clarify") as resolve:
        await adapter._on_reaction(_reaction_event("1️⃣", sender=bob))

    resolve.assert_not_called()
    assert "$clarify-prompt" in adapter._clarify_prompts_by_event
    adapter._send_invalid_reaction_feedback.assert_awaited_once_with(
        ROOM_ID,
        "$clarify-prompt",
        "Only the user who requested this action can use these controls.",
    )


@pytest.mark.asyncio
async def test_duplicate_reaction_event_resolves_at_most_once(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick",
        choices=["Alpha"],
        clarify_id="clarify-duplicate",
        session_key="matrix:duplicate",
    )
    adapter._redact_bot_clarify_reactions = AsyncMock()
    event = _reaction_event("1️⃣", event_id="$same-reaction")

    with patch(
        "tools.clarify_gateway.resolve_gateway_clarify", return_value=True
    ) as resolve:
        await adapter._on_reaction(event)
        await adapter._on_reaction(event)

    resolve.assert_called_once_with("clarify-duplicate", "Alpha")


@pytest.mark.asyncio
async def test_typed_response_wins_over_late_reaction_and_cleans_prompt(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    entry = clarify_gateway.register("clarify-stale", "matrix:stale", "Pick", ["Alpha"])
    await adapter.send_clarify(
        chat_id=ROOM_ID,
        question="Pick",
        choices=["Alpha"],
        clarify_id="clarify-stale",
        session_key="matrix:stale",
    )
    adapter._redact_bot_clarify_reactions = AsyncMock()
    adapter._send_invalid_reaction_feedback = AsyncMock()

    assert (
        clarify_gateway.resolve_text_response_for_session(
            "matrix:stale", "typed response"
        )
        is True
    )
    await adapter._on_reaction(_reaction_event("1️⃣"))

    assert entry.response == "typed response"
    assert "$clarify-prompt" not in adapter._clarify_prompts_by_event
    adapter._redact_bot_clarify_reactions.assert_awaited_once()
    adapter._send_invalid_reaction_feedback.assert_awaited_once_with(
        ROOM_ID,
        "$clarify-prompt",
        "This clarify prompt is no longer active.",
    )


@pytest.mark.asyncio
async def test_clarify_expiry_uses_gateway_timeout(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    before = time.monotonic()

    with patch("tools.clarify_gateway.get_clarify_timeout", return_value=17):
        await adapter.send_clarify(
            chat_id=ROOM_ID,
            question="Pick",
            choices=["Alpha"],
            clarify_id="clarify-timeout",
            session_key="matrix:timeout",
        )

    prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]
    assert before + 17 <= prompt.expires_at <= time.monotonic() + 17


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id", [None, "$thread"])
@pytest.mark.parametrize("outcome", ["reaction", "typed", "timeout", "cancel"])
async def test_real_gateway_callback_binds_requester_and_retires_controls(
    monkeypatch, thread_id, outcome
):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext
    from gateway.run_turn_runner import TurnRunner

    adapter = _make_adapter(monkeypatch)
    adapter._allowed_user_ids.add("@bob:example.org")
    adapter._schedule_reaction_redaction = Mock()
    monkeypatch.setattr(clarify_gateway, "get_clarify_timeout", lambda: 1)
    posted = asyncio.Event()
    original = adapter.send_clarify

    async def send_prompt(**kwargs):
        result = await original(**kwargs)
        posted.set()
        return result

    adapter.send_clarify = send_prompt
    metadata = {"thread_id": thread_id} if thread_id else None
    source = SessionSource(
        platform=Platform.MATRIX,
        chat_id=ROOM_ID,
        user_id=REQUESTER,
        thread_id=thread_id,
    )
    ctx = TurnContext(
        source=source,
        session_key="matrix:integration",
        _status_adapter=adapter,
        _status_chat_id=ROOM_ID,
        _status_thread_metadata=metadata,
        _loop_for_step=asyncio.get_running_loop(),
    )
    task = asyncio.create_task(
        asyncio.to_thread(
            TurnRunner(None, ctx)._clarify_callback_sync,
            "Pick",
            ["✋ Stop", "Continue"],
        )
    )
    try:
        await asyncio.wait_for(posted.wait(), 5)
        prompt = adapter._clarify_prompts_by_event["$clarify-prompt"]
        assert prompt.requester_user_id == REQUESTER
        assert metadata == ({"thread_id": thread_id} if thread_id else None)
        await adapter._on_reaction(
            _reaction_event("✋", sender="@bob:example.org", event_id="$wrong-user")
        )
        assert not clarify_gateway._entries[prompt.clarify_id].event.is_set()
        if outcome == "reaction":
            await adapter._on_reaction(_reaction_event("✋️"))
        elif outcome == "typed":
            assert clarify_gateway.resolve_text_response_for_session(
                ctx.session_key, "2"
            )
        elif outcome == "cancel":
            clarify_gateway.clear_session(ctx.session_key)
        result = await asyncio.wait_for(task, 5)
        assert (
            result
            == {
                "reaction": "✋ Stop",
                "typed": "Continue",
                "timeout": "[user did not respond within 0m]",
                "cancel": "[user did not respond within 0m]",
            }[outcome]
        )
        assert not adapter._clarify_prompts_by_event
        assert not adapter._clarify_prompt_event_by_id
        assert {
            c.args[1] for c in adapter._schedule_reaction_redaction.call_args_list
        } == set(prompt.bot_reaction_events.values())
    finally:
        clarify_gateway.clear_session(ctx.session_key)
        await asyncio.wait_for(task, 5)


def test_resolved_entry_cannot_reenter_text_mode_or_mask_next_request():
    first = clarify_gateway.register("first", "session", "Question", ["Alpha"])
    second = clarify_gateway.register("second", "session", "Question", None)
    assert clarify_gateway.resolve_gateway_clarify("first", "Alpha")
    assert not clarify_gateway.mark_awaiting_text("first")
    assert (
        clarify_gateway.get_pending_for_session("session", include_choice_prompts=True)
        is second
    )
    assert first.response == "Alpha"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["unlimited", "seed_failure", "answer_during_seed"])
async def test_seed_lifecycle_boundaries(monkeypatch, mode):
    adapter = _make_adapter(monkeypatch)
    adapter._schedule_reaction_redaction = Mock()
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True))
    entry = clarify_gateway.register(
        "boundary", "boundary-session", "Pick", ["Alpha", "Beta"]
    )
    if mode == "unlimited":
        monkeypatch.setattr(clarify_gateway, "get_clarify_timeout", lambda: 0)
    if mode == "seed_failure":
        adapter._send_reaction = AsyncMock(return_value=None)
    if mode == "answer_during_seed":

        async def seed(room, event, emoji):
            if emoji == "1️⃣":
                await adapter._on_reaction(_reaction_event(emoji))
            return "$seed-" + emoji

        adapter._send_reaction = AsyncMock(side_effect=seed)
    await adapter.send_clarify(
        ROOM_ID, "Pick", ["Alpha", "Beta"], "boundary", "boundary-session"
    )
    if mode == "unlimited":
        assert adapter._clarify_prompts_by_event["$clarify-prompt"].expires_at is None
    elif mode == "seed_failure":
        assert not adapter._clarify_prompts_by_event
        assert (
            "React to choose" not in adapter.edit_message.await_args.kwargs["content"]
        )
    else:
        assert entry.response == "Alpha"
        assert not adapter._clarify_prompts_by_event
        assert {
            c.args[1] for c in adapter._schedule_reaction_redaction.call_args_list
        } == {"$seed-1️⃣"}


@pytest.mark.asyncio
@pytest.mark.parametrize("seed_failure", [False, True])
async def test_actual_matrix_transport_gets_thread_message_and_annotation_events(
    monkeypatch, seed_failure
):
    from plugins.platforms.matrix.adapter import EventType

    adapter = _make_adapter(monkeypatch)
    del adapter.send
    del adapter._send_reaction
    adapter._schedule_reaction_redaction = Mock()
    events = []

    async def send_event(room, kind, content):
        if kind == EventType.REACTION and seed_failure:
            raise RuntimeError("test transport unavailable")
        events.append((str(room), kind, content))
        return "$clarify-prompt" if len(events) == 1 else f"$sent-{len(events)}"

    adapter._client = SimpleNamespace(send_message_event=send_event)
    clarify_gateway.register(
        "transport", "transport-session", "Pick", ["🚀 Execute", "Pause"]
    )
    result = await adapter.send_clarify(
        ROOM_ID,
        "Pick",
        ["🚀 Execute", "Pause"],
        "transport",
        "transport-session",
        metadata={"thread_id": "$thread", "requester_user_id": REQUESTER},
    )
    assert result.success
    assert events[0][2]["m.relates_to"]["event_id"] == "$thread"
    if seed_failure:
        assert not adapter._clarify_prompts_by_event
        assert events[-1][2]["m.relates_to"] == {
            "rel_type": "m.replace",
            "event_id": "$clarify-prompt",
        }
        assert "unavailable" in events[-1][2]["m.new_content"]["body"]
    else:
        assert [event[2]["m.relates_to"] for event in events[1:]] == [
            {"rel_type": "m.annotation", "event_id": "$clarify-prompt", "key": key}
            for key in ("🚀", "2️⃣", OTHER)
        ]
        assert all(event[1] == EventType.REACTION for event in events[1:])
